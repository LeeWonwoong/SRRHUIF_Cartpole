import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import os
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('Agg')

"""
=========================================================================
SRRHUIF-D3QN v8.0: Full Horizon Log + Diagnostic Edition
=========================================================================
v7 fulldiag 기반 + 진단 로그 추가:
  [호라이즌 단위 (h=0..N-1)]  ← 기존 ||H^T||, ||Δθ||, cos(δ) 등과 함께
    - cond(Y) per layer: S 대각 기반 pseudo condition number
    - Y_max per layer: max eigenvalue 근사
  
  [에피소드 단위]
    - ||θ|| per layer: 파라미터 norm 진화
    - Adv null/signal ratio: Dueling identifiability 지표
    - Shared eff_rank + stable_rank: Feature collapse 지표
    - Argmax flip rate: Update-induced policy instability
    - Reference state ΔQ: 5개 고정 state에서 Q(right)-Q(left)
  
  [옵션]
    - --use_full_eigvalsh: 정확한 condition number (느림)
=========================================================================
"""

print("=" * 70)
print(f"SRRHUIF-D3QN v8.0 (FullDiag) | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print("=" * 70)

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
DTYPE_FWD = torch.float32
JITTER = 1e-10

# =========================================================================
# 1. Configuration (업로드 파일 기준 설정 유지)
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_episodes: int = 120
    max_steps: int = 500
    batch_size: int = 64
    buffer_size: int = 20000

    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    gamma: float = 0.9
    scale_factor: float = 1.0
    
    tau_srrhuif: float = 0.02
    N_horizon: int = 9
    q_std: float = 5e-4
    r_std: float = 1.8

    alpha: float = 0.3
    beta: float = 2.0   
    kappa: float = 0.0
    
    max_k_gain: float = 0.0
    
    p_init: float = 0.03
    value_layer_scale: float = 1
    use_spas: bool = True  

    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000

    update_interval: int = 4
    use_input_norm: bool = True
    use_compile: bool = True
    plot_interval: int = 40
    log_interval : int = 1
    seed: int = 0
    
    # === DIAGNOSTIC FLAGS ===
    use_full_eigvalsh: bool = True  # True시 정확한 cond 계산 (느림)
    diag_ref_states: bool = True         # Q_ref_state 추적
    diag_argmax_flip: bool = True        # argmax flip rate
    diag_eff_rank: bool = True           # shared output effective rank
    diag_horizon_cond: bool = True       # 호라이즌별 cond/Y_max

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_std
        self.r_inv = 1.0 / (self.r_std ** 2)

        self.param_str = f"a{self.alpha}_b{self.beta}_r{self.r_std}_p{self.p_init}_vs{self.value_layer_scale}"
        self.outdir = f"./results_cartpole/{self.param_str}"
        os.makedirs(self.outdir, exist_ok=True)


cfg = Config()

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=cfg.alpha)
parser.add_argument('--beta', type=float, default=cfg.beta)
parser.add_argument('--r_std', type=float, default=cfg.r_std)
parser.add_argument('--p_init', type=float, default=cfg.p_init)
parser.add_argument('--episodes', type=int, default=cfg.max_episodes)
parser.add_argument('--horizon', type=int, default=cfg.N_horizon)
parser.add_argument('--batch', type=int, default=cfg.batch_size)
parser.add_argument('--q_std', type=float, default=cfg.q_std)
parser.add_argument('--tau', type=float, default=cfg.tau_srrhuif)
parser.add_argument('--value_layer_scale', type=float, default=cfg.value_layer_scale) 
parser.add_argument('--use_full_eigvalsh', action='store_true', default=cfg.use_full_eigvalsh)
args, _ = parser.parse_known_args()

cfg.alpha = args.alpha
cfg.beta = args.beta
cfg.r_std = args.r_std
cfg.p_init = args.p_init
cfg.max_episodes = args.episodes
cfg.N_horizon = args.horizon
cfg.batch_size = args.batch
cfg.q_std = args.q_std
cfg.tau_srrhuif = args.tau
cfg.value_layer_scale = args.value_layer_scale
cfg.use_full_eigvalsh = args.use_full_eigvalsh

cfg.__post_init__()

# =========================================================================
# 2. Network Info & Cache
# =========================================================================
def create_network_info(dimS: int, nA: int, config: Config) -> Dict:
    info = {'dimS': dimS, 'nA': nA, 'layers': [], 'nd_layers': []}
    idx, ld_idx = 0, 0
    def add_layers(sizes, type_str):
        nonlocal idx, ld_idx
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            layer = {
                'type': type_str, 'layer_idx': i,
                'W_start': idx, 'W_len': fan_out * fan_in, 'W_shape': (fan_out, fan_in),
                'b_start': idx + fan_out * fan_in, 'b_len': fan_out,
                'fan_in': fan_in, 'fan_out': fan_out,
            }
            idx += fan_out * fan_in + fan_out
            info['layers'].append(layer)
            info['nd_layers'].append({
                'global_idx': ld_idx, 'type': type_str, 'local_idx': i,
                'fan_in': fan_in, 'fan_out': fan_out, 'n_per_neuron': fan_in + 1,
                'W_start': layer['W_start'], 'W_len': layer['W_len'],
                'b_start': layer['b_start'], 'b_len': layer['b_len']})
            ld_idx += 1
    shared_out = config.shared_layers[-1]
    add_layers([dimS] + config.shared_layers, 'shared')
    info['shared_end_idx'] = len(info['layers'])
    add_layers([shared_out] + config.value_layers + [1], 'value')
    info['value_end_idx'] = len(info['layers'])
    add_layers([shared_out] + config.advantage_layers + [nA], 'advantage')
    info['total_params'] = idx
    info['num_nd_layers'] = len(info['nd_layers'])
    return info

class NDCache:
    def __init__(self, info: Dict, cfg: Config, device: str):
        self.layers = {}
        total_forwards = 0
        layer_fwd_slices = []
        for L, nd_layer in enumerate(info['nd_layers']):
            fan_in, fan_out = nd_layer['fan_in'], nd_layer['fan_out']
            n_per = nd_layer['n_per_neuron']
            num_sigma = 2 * n_per + 1
            W_start, b_start = nd_layer['W_start'], nd_layer['b_start']
            count = fan_out * num_sigma
            layer_fwd_slices.append((total_forwards, total_forwards + count))
            total_forwards += count
            j_idx = torch.arange(fan_out, device=device).view(-1, 1, 1)
            k_idx = torch.arange(fan_in, device=device).view(1, 1, -1)
            w_col_idx = (W_start + j_idx * fan_in + k_idx).expand(fan_out, num_sigma, fan_in).contiguous()
            b_col_idx = (b_start + j_idx.squeeze(-1)).expand(fan_out, num_sigma).unsqueeze(-1).contiguous()
            eye_n_per = torch.eye(n_per, dtype=DTYPE, device=device)
            eye_n_per_batch = eye_n_per.unsqueeze(0).expand(fan_out, -1, -1).clone()
            lamb = cfg.alpha ** 2 * (n_per + cfg.kappa) - n_per
            gamma = float(np.sqrt(n_per + lamb))
            Wm = torch.zeros(2 * n_per + 1, dtype=DTYPE, device=device)
            Wc = torch.zeros(2 * n_per + 1, dtype=DTYPE, device=device)
            Wm[0] = lamb / (n_per + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha ** 2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (n_per + lamb)
            S_Q_cached = cfg.q_std * eye_n_per_batch.clone()
            Wm_col_f32 = Wm.to(DTYPE_FWD).view(1, -1, 1).expand(fan_out, -1, -1).clone()
            Wc_f32 = Wc.to(DTYPE_FWD)
            zero_col_f32 = torch.zeros(fan_out, n_per, 1, dtype=DTYPE_FWD, device=device)
            self.layers[L] = {
                'w_col_idx': w_col_idx, 'b_col_idx': b_col_idx,
                'eye_n_per': eye_n_per, 'eye_n_per_batch': eye_n_per_batch,
                'Wm': Wm, 'Wc': Wc, 'gamma': gamma,
                'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per, 'num_sigma': num_sigma,
                'S_Q_cached': S_Q_cached,
                'Wm_col_f32': Wm_col_f32, 'Wc_f32': Wc_f32, 'zero_col_f32': zero_col_f32,
            }
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards

        self.n_per_groups = {}
        for L, nd_layer in enumerate(info['nd_layers']):
            n_per = nd_layer['n_per_neuron']
            if n_per not in self.n_per_groups:
                self.n_per_groups[n_per] = {
                    'layers': [], 'fan_outs': [], 'total_neurons': 0
                }
            grp = self.n_per_groups[n_per]
            grp['layers'].append(L)
            grp['fan_outs'].append(nd_layer['fan_out'])
            grp['total_neurons'] += nd_layer['fan_out']
        
        for n_per, grp in self.n_per_groups.items():
            total_n = grp['total_neurons']
            grp['eye_grouped'] = torch.eye(n_per, dtype=DTYPE, device=device).unsqueeze(0).expand(total_n, -1, -1).clone()
            grp['S_Q_grouped'] = cfg.q_std * grp['eye_grouped'].clone()
            grp['gamma'] = self.layers[grp['layers'][0]]['gamma']
            offsets = [0]
            for fo in grp['fan_outs']:
                offsets.append(offsets[-1] + fo)
            grp['offsets'] = offsets

    def get(self, layer_idx: int) -> Dict:
        return self.layers[layer_idx]

class InputNormalizer:
    def __init__(self, device):
        self.scale = torch.tensor([2.4, 3.0, 0.21, 2.0], dtype=DTYPE, device=device)
    def normalize(self, x):
        if x.dim() == 1: return x / self.scale
        elif x.shape[-1] == len(self.scale): return x / self.scale
        else: return x / self.scale.view(-1, 1)

# =========================================================================
# 3. Forward Functions & Replay Buffer & Math Utils
# =========================================================================
def forward_single(theta, info, x):
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        h = F.relu(W @ h + b)
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = W @ v + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = W @ a + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    return (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)

def forward_single_with_shared(theta, info, x):
    """(Q, shared_output) 반환 — effective rank 계산용"""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        h = F.relu(W @ h + b)
    shared_out = h.clone()
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = W @ v + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = W @ a + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    Q = (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    return Q, shared_out.to(DTYPE)

def forward_bmm(thetas, info, x):
    thetas = thetas.to(DTYPE_FWD); x = x.to(DTYPE_FWD)
    num_sigma = thetas.shape[0]
    x_expanded = x.t().unsqueeze(0).expand(num_sigma, -1, -1)
    h = x_expanded
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        h = F.relu(torch.bmm(W, h) + b)
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, v) + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, a) + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    return v + (a - a.mean(dim=1, keepdim=True))

class TensorReplayBuffer:
    def __init__(self, capacity: int, dimS: int, device: str):
        self.capacity, self.count, self.device = capacity, 0, device
        self.S = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term = torch.zeros(capacity, dtype=DTYPE, device=device)

    def push(self, s, a, r, s_next, done):
        idx = self.count % self.capacity
        self.S[idx] = torch.as_tensor(s, dtype=DTYPE, device=self.device)
        self.A[idx] = a; self.R[idx] = r
        self.S_next[idx] = torch.as_tensor(s_next, dtype=DTYPE, device=self.device)
        self.term[idx] = float(done); self.count += 1

    @property
    def current_size(self): return min(self.count, self.capacity)

    def sample_batch(self, batch_size: int) -> Dict:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return {'s': self.S[indices].t(), 'a': self.A[indices], 'r': self.R[indices],
                's_next': self.S_next[indices].t(), 'term': self.term[indices]}

def tria_operation_batch(A):
    _, r = torch.linalg.qr(A.transpose(-2, -1).contiguous())
    s = r.transpose(-2, -1).contiguous()
    d = torch.diagonal(s, dim1=-2, dim2=-1)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    return s * signs.unsqueeze(-1)

def safe_inv_tril_batch(L_batch, eye_batch):
    result = torch.linalg.solve_triangular(L_batch + JITTER * eye_batch, eye_batch, upper=False)
    return torch.where(torch.isfinite(result), result, eye_batch)

def robust_solve_spd_batch(S_tril_batch, y_batch, eye_batch):
    S_safe = S_tril_batch + JITTER * eye_batch
    z = torch.linalg.solve_triangular(S_safe, y_batch, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.transpose(-2, -1).contiguous(), z, upper=True)
    return torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))

# =========================================================================
# DIAGNOSTIC UTILITIES
# =========================================================================
@torch.no_grad()
def compute_pseudo_cond_from_S(S_batch):
    """
    S의 대각 기반 pseudo condition number.
    S가 lower triangular이고 Y = S^-T S^-1 이므로 Y 대각 ≈ 1/S_diag²
    거의 공짜.
    Returns: (cond_mean, y_max_over_batch, y_min_over_batch)
    """
    S_diag = torch.diagonal(S_batch, dim1=-2, dim2=-1).abs().clamp(min=1e-12)
    Y_diag = 1.0 / (S_diag ** 2)  # (neurons, n_per)
    y_max_per_neuron = Y_diag.max(dim=-1).values
    y_min_per_neuron = Y_diag.min(dim=-1).values
    cond_per_neuron = y_max_per_neuron / y_min_per_neuron.clamp(min=1e-12)
    return (cond_per_neuron.mean().item(),
            y_max_per_neuron.max().item(),
            y_min_per_neuron.min().item())

@torch.no_grad()
def compute_full_cond_from_S(S_batch):
    """Full eigenvalue 기반 정확한 cond (비용 높음)"""
    try:
        SST = torch.bmm(S_batch, S_batch.transpose(-2, -1))
        eigvals_SST = torch.linalg.eigvalsh(SST)  # ascending
        y_max = 1.0 / eigvals_SST[:, 0].clamp(min=1e-12)
        y_min = 1.0 / eigvals_SST[:, -1].clamp(min=1e-12)
        cond = y_max / y_min.clamp(min=1e-12)
        return cond.mean().item(), y_max.max().item()
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_effective_rank(X, tol_ratio=1e-3):
    """X: (dim, batch) or (batch, dim). SVD 기반 effective rank + stable rank."""
    if X.shape[0] > X.shape[1]:
        X = X.t()
    try:
        X_centered = X - X.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(X_centered)
        s_max = s.max()
        if s_max < 1e-12:
            return 0.0, 0.0
        eff_rank = (s > s_max * tol_ratio).sum().item()
        stable_rank = (s ** 2).sum().item() / (s_max ** 2).item()
        return float(eff_rank), float(stable_rank)
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_advantage_null_ratio(theta, info):
    """A1 (advantage 마지막 레이어)의 null vs signal component"""
    adv_layers = [L for L in info['nd_layers'] if L['type'] == 'advantage']
    if not adv_layers:
        return 0.0, 0.0, 0.0
    a1_layer = adv_layers[-1]
    W_start, W_len = a1_layer['W_start'], a1_layer['W_len']
    b_start, b_len = a1_layer['b_start'], a1_layer['b_len']
    fan_in, fan_out = a1_layer['fan_in'], a1_layer['fan_out']
    
    theta_flat = theta.squeeze()
    W = theta_flat[W_start:W_start + W_len].view(fan_out, fan_in)
    b = theta_flat[b_start:b_start + b_len]
    
    # null: 모든 action 뉴런이 같은 방향. signal: 뉴런간 deviation
    W_mean = W.mean(dim=0)          # (fan_in,) per-input null component
    W_dev = W - W_mean.unsqueeze(0) # (nA, fan_in)
    
    null_norm = W_mean.norm().item()
    signal_norm = W_dev.norm().item()
    b_mean = b.mean().item()
    b_dev_norm = (b - b.mean()).norm().item()
    
    null_total = (null_norm ** 2 + b_mean ** 2) ** 0.5
    signal_total = (signal_norm ** 2 + b_dev_norm ** 2) ** 0.5
    ratio = null_total / (signal_total + 1e-12)
    return ratio, null_total, signal_total

@torch.no_grad()
def compute_layer_theta_norms(theta, info):
    """Layer별 parameter norm dict"""
    norms = {}
    theta_flat = theta.squeeze()
    for L, nd_layer in enumerate(info['nd_layers']):
        ltype = nd_layer['type']
        lidx = nd_layer['local_idx']
        label = f"{ltype[0].upper()}{lidx}"
        W_start, W_len = nd_layer['W_start'], nd_layer['W_len']
        b_start, b_len = nd_layer['b_start'], nd_layer['b_len']
        W_norm = theta_flat[W_start:W_start + W_len].norm().item()
        b_norm = theta_flat[b_start:b_start + b_len].norm().item()
        norms[label] = (W_norm ** 2 + b_norm ** 2) ** 0.5
    return norms

# CartPole 고정 reference states
REF_STATES = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],       # 완전 균형
    [0.0, 0.0, 0.05, 0.0],      # 약간 기울어짐 (R)
    [0.0, 0.0, -0.05, 0.0],     # 약간 기울어짐 (L)
    [0.0, 0.0, 0.1, 0.5],       # 떨어지는 중 (R)
    [0.0, 0.0, -0.1, -0.5],     # 떨어지는 중 (L)
], dtype=DTYPE)
REF_NAMES = ["balance", "tilt_R", "tilt_L", "fall_R", "fall_L"]

@torch.no_grad()
def compute_ref_q_values(theta, info, normalizer, device):
    """5개 reference state에서 Q, ΔQ 계산"""
    ref = REF_STATES.to(device)
    ref_norm = normalizer.normalize(ref) if normalizer else ref
    Q = forward_single(theta.squeeze(), info, ref_norm.t())  # (nA, n_refs)
    results = {}
    for i, name in enumerate(REF_NAMES):
        q0 = Q[0, i].item()
        q1 = Q[1, i].item()
        dq = q1 - q0
        argmax = 1 if dq > 0 else 0
        results[name] = {'q0': q0, 'q1': q1, 'dq': dq, 'argmax': argmax}
    return results

# =========================================================================
# 6. ND Core Functions
# =========================================================================
def _nd_time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    combined = torch.cat([P_sqrt_prev, S_Q_cached], dim=2)
    P_sqrt_pred = tria_operation_batch(combined)
    S_pred = safe_inv_tril_batch(P_sqrt_pred, eye_batch)
    Y_pred = torch.bmm(S_pred, S_pred.transpose(-2, -1))
    y_pred = torch.bmm(Y_pred, theta_3d)
    scaled_P = gamma_val * P_sqrt_pred
    theta_2d = theta_3d.squeeze(-1)
    X_sigma_all = torch.cat([
        theta_2d.unsqueeze(1),
        theta_2d.unsqueeze(1) + scaled_P.transpose(-2, -1),
        theta_2d.unsqueeze(1) - scaled_P.transpose(-2, -1),
    ], dim=1)
    return S_pred, Y_pred, y_pred, X_sigma_all, scaled_P

def _nd_compute_ht_core(Z_sigma_T_f32, Wm_col_f32, Wc_f32, zero_col_f32,
                         scaled_P_f32, z_measured_exp_f64, Y_pred_f64):
    z_hat_f32 = torch.bmm(Z_sigma_T_f32, Wm_col_f32)
    Z_dev_f32 = Z_sigma_T_f32 - z_hat_f32
    X_dev_f32 = torch.cat([zero_col_f32, scaled_P_f32, -scaled_P_f32], dim=2)
    P_xz_f32 = torch.bmm(X_dev_f32 * Wc_f32.view(1, 1, -1), Z_dev_f32.transpose(1, 2))
    z_hat_f64 = z_hat_f32.to(torch.float64)
    residual_all = z_measured_exp_f64 - z_hat_f64
    HT_all = torch.bmm(Y_pred_f64, P_xz_f32.to(torch.float64))
    
    ht_norm = torch.norm(HT_all, dim=1).mean().item()
    resid_norm = torch.norm(residual_all, dim=1).mean().item()
    
    return HT_all, residual_all, z_hat_f64, ht_norm, resid_norm

def _nd_meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all, r_inv_sqrt, r_inv, eye_batch):
    combined = torch.cat([S_pred, HT_all * r_inv_sqrt], dim=2)
    S_new_all = tria_operation_batch(combined)
    
    ht_theta = torch.bmm(HT_all.transpose(1, 2), theta_3d)
    innov = residual_all + ht_theta
    
    innov_abs = torch.abs(innov)
    innov_mean = torch.mean(innov_abs)
    innov_max = torch.max(innov_abs)
    resid_in_innov = torch.mean(torch.abs(residual_all)).item()
    ht_theta_in_innov = torch.mean(torch.abs(ht_theta)).item()
    innov_norm = innov_mean.item()
    
    y_new_all = y_pred + torch.bmm(HT_all, r_inv * innov)
    
    delta_y = torch.bmm(HT_all, r_inv * innov)
    delta_y_norm = torch.norm(delta_y, dim=1).mean()
    y_pred_norm = torch.norm(y_pred, dim=1).mean().item()
    y_new_norm = torch.norm(y_new_all, dim=1).mean()
    
    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    
    S_diag = torch.diagonal(S_new_all, dim1=-2, dim2=-1)
    P_approx = 1.0 / (S_diag ** 2 + 1e-12)
    avg_P_new = P_approx.mean().item()
    
    meas_stats = {
        'innov_mean': innov_mean, 'innov_max': innov_max,
        'resid_in_innov': resid_in_innov,
        'ht_theta_in_innov': ht_theta_in_innov,
        'innov_norm': innov_norm,
        'delta_y': delta_y_norm.item(),
        'y_pred_norm': y_pred_norm,
        'y_new_norm': y_new_norm.item(),
        'avg_P': avg_P_new,
    }
    return theta_new_all, S_new_all, meas_stats

# =========================================================================
# 7. Initialize theta & SRRHUIF ND Step
# =========================================================================
def initialize_theta(info, device):
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    for layer in info['layers']:
        fan_in, W_len = layer['W_shape'][1], layer['W_len']
        theta[layer['W_start']:layer['W_start'] + W_len] = \
            torch.randn(W_len, dtype=DTYPE, device=device) * np.sqrt(2.0 / fan_in)
    return theta

@torch.no_grad()
def srrhuif_step_nd(theta_current_in, theta_target, neuron_S_info, batch, sp,
                     is_first, p_init_val, nd_cache, q_next_target_cached=None):
    """
    SRRHUIF-ND step with per-horizon layer-wise cond/Y_max diagnostics.
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info_dict = {}
    total_loss, layer_count = 0.0, 0

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    s_batch = s_batch + torch.randn_like(s_batch) * 0.02
    unified = nd_cache.unified_thetas
    unified[:] = theta_current.squeeze().to(DTYPE_FWD)

    # Phase 1: Time Update
    per_layer = {}
    for L in range(info['num_nd_layers']):
        nd_layer = info['nd_layers'][L]
        lc = nd_cache.get(L)
        fan_in, fan_out, n_per = lc['fan_in'], lc['fan_out'], lc['n_per']
        W_start, b_start = nd_layer['W_start'], nd_layer['b_start']

        W_prior = theta_prior.squeeze()[W_start:W_start + nd_layer['W_len']].view(fan_out, fan_in)
        b_prior = theta_prior.squeeze()[b_start:b_start + nd_layer['b_len']]
        theta_all_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1)
        theta_all_prior_3d = theta_all_prior.unsqueeze(-1)

        S_3d = neuron_S_info[L]
        if is_first or S_3d is None:
            current_p_init = p_init_val
            if cfg.value_layer_scale != 1.0 and ('value' in nd_layer['type'] or 'advantage' in nd_layer['type']):
                current_p_init = p_init_val * cfg.value_layer_scale
            P_sqrt_prev = np.sqrt(current_p_init) * lc['eye_n_per_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_n_per_batch'])

        per_layer[L] = {
            'nd_layer': nd_layer, 'lc': lc,
            'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per,
            'W_start': W_start, 'b_start': b_start,
            'theta_all_prior': theta_all_prior,
            'theta_all_prior_3d': theta_all_prior_3d,
            'P_sqrt_prev': P_sqrt_prev,
        }

    for n_per_val, grp in nd_cache.n_per_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev'] for L in layers_in_grp], dim=0)

        S_pred_g, Y_pred_g, y_pred_g, X_sigma_g, scaled_P_g = _nd_time_update_core(
            all_theta_3d, all_P_sqrt, grp['S_Q_grouped'], grp['eye_grouped'], grp['gamma'])

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            per_layer[L]['S_pred'] = S_pred_g[s:e]
            per_layer[L]['Y_pred'] = Y_pred_g[s:e]
            per_layer[L]['y_pred'] = y_pred_g[s:e]
            per_layer[L]['X_sigma_all'] = X_sigma_g[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]

    for L in range(info['num_nd_layers']):
        pl = per_layer[L]
        lc = pl['lc']
        X_sigma_f32 = pl['X_sigma_all'].to(DTYPE_FWD)
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(pl['fan_out'], lc['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :pl['fan_in']])
        layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, pl['fan_in']:pl['fan_in'] + 1])

    # Phase 2: z_measured
    if q_next_target_cached is not None:
        if is_first and cfg.use_spas:
            Q_sigma_f32 = forward_bmm(unified, info, s_next)
            a_best_next = Q_sigma_f32.mean(dim=0).argmax(dim=0)
        else:
            Q_curr = forward_single(theta_current.squeeze(), info, s_next)
            a_best_next = Q_curr.argmax(dim=0)
        q_val_next = q_next_target_cached[a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)
    else:
        if is_first:
            if cfg.use_spas:
                Q_sigma_f32 = forward_bmm(unified, info, s_next)
                a_best_next = Q_sigma_f32.mean(dim=0).argmax(dim=0)
                Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)
                q_val_next = Q_tgt_f32[0][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)
            else:
                Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)
                q_val_next = Q_tgt_f32[0].max(dim=0).values.to(DTYPE)
        else:
            thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
            Q_both_f32 = forward_bmm(thetas_pair, info, s_next)
            a_best_next = Q_both_f32[0].argmax(dim=0)
            q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)

    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    target_var = torch.var(z_measured).item()

    # Phase 3: H computation
    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    for L in range(info['num_nd_layers']):
        pl = per_layer[L]
        lc = pl['lc']
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(pl['fan_out'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
        z_measured_exp = z_measured.unsqueeze(0).expand(pl['fan_out'], -1, -1)

        HT_all, residual_all, z_hat_f64, ht_norm, resid_norm = _nd_compute_ht_core(
            Z_sigma_T_f32, lc['Wm_col_f32'], lc['Wc_f32'], lc['zero_col_f32'],
            pl['scaled_P'].to(DTYPE_FWD), z_measured_exp, pl['Y_pred'])

        per_layer[L]['HT_all'] = HT_all
        per_layer[L]['residual_all'] = residual_all
        per_layer[L]['loss'] = torch.mean(residual_all ** 2)
        per_layer[L]['ht_norm'] = ht_norm
        per_layer[L]['resid_norm'] = resid_norm
        layer_count += 1

    # Phase 4: Measurement Update + PER-LAYER COND DIAGNOSTICS
    total_innov_mean = 0.0
    total_innov_max = 0.0
    total_ht_norm = 0.0
    total_resid_norm = 0.0
    total_delta_y = 0.0
    total_y_new = 0.0
    total_avg_P = 0.0
    total_resid_in_innov = 0.0
    total_ht_theta_in_innov = 0.0
    total_innov_norm = 0.0
    total_y_pred_norm = 0.0
    group_count = 0
    
    # === DIAGNOSTIC: per-layer cond/Y_max ===
    per_layer_cond = {}
    per_layer_ymax = {}
    per_layer_cond_full = {} if cfg.use_full_eigvalsh else None

    for n_per_val, grp in nd_cache.n_per_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_S_pred = torch.cat([per_layer[L]['S_pred'] for L in layers_in_grp], dim=0)
        all_y_pred = torch.cat([per_layer[L]['y_pred'] for L in layers_in_grp], dim=0)
        all_HT = torch.cat([per_layer[L]['HT_all'] for L in layers_in_grp], dim=0)
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_residual = torch.cat([per_layer[L]['residual_all'] for L in layers_in_grp], dim=0)

        theta_new_g, S_new_g, meas_stats = _nd_meas_update_core(
            all_S_pred, all_y_pred, all_HT, all_theta_3d,
            all_residual, cfg.r_inv_sqrt, cfg.r_inv, grp['eye_grouped'])

        total_innov_mean += meas_stats['innov_mean'].item()
        total_innov_max = max(total_innov_max, meas_stats['innov_max'].item())
        total_delta_y += meas_stats['delta_y']
        total_y_new += meas_stats['y_new_norm']
        total_avg_P += meas_stats['avg_P']
        total_resid_in_innov += meas_stats['resid_in_innov']
        total_ht_theta_in_innov += meas_stats['ht_theta_in_innov']
        total_innov_norm += meas_stats['innov_norm']
        total_y_pred_norm += meas_stats['y_pred_norm']
        
        for L in layers_in_grp:
            total_ht_norm += per_layer[L]['ht_norm']
            total_resid_norm += per_layer[L]['resid_norm']
        
        group_count += 1

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            pl = per_layer[L]
            theta_new_L = theta_new_g[s:e]
            S_new_L = S_new_g[s:e]
            
            # === DIAGNOSTIC: cond/Y_max per layer ===
            if cfg.diag_horizon_cond:
                nd = info['nd_layers'][L]
                label = f"{nd['type'][0].upper()}{nd['local_idx']}"
                cond_val, ymax_val, ymin_val = compute_pseudo_cond_from_S(S_new_L)
                per_layer_cond[label] = cond_val
                per_layer_ymax[label] = ymax_val
                
                if cfg.use_full_eigvalsh:
                    full_cond, full_ymax = compute_full_cond_from_S(S_new_L)
                    per_layer_cond_full[label] = full_cond

            invalid = ~torch.isfinite(theta_new_L).all(dim=(1, 2))
            if invalid.any():
                theta_new_L[invalid] = pl['theta_all_prior'][invalid].unsqueeze(-1)

            W_new = theta_new_L[:, :pl['fan_in'], 0]
            b_new = theta_new_L[:, pl['fan_in'], 0]
            theta_flat = theta_current.squeeze()
            theta_flat[pl['W_start']:pl['W_start'] + pl['nd_layer']['W_len']] = W_new.reshape(-1)
            theta_flat[pl['b_start']:pl['b_start'] + pl['nd_layer']['b_len']] = b_new
            theta_current = theta_flat.view(-1, 1)

            new_S_info_dict[L] = S_new_L.permute(1, 2, 0)
            total_loss = total_loss + pl['loss']

    new_S_info = [new_S_info_dict[L] for L in range(info['num_nd_layers'])]
    
    delta_theta = theta_current.squeeze() - theta_current_in.squeeze()
    k_gain_norm = torch.norm(delta_theta).item()
    
    if cfg.max_k_gain > 0 and k_gain_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / k_gain_norm
        theta_current = (theta_current_in.squeeze() + delta_theta * scale).view(-1, 1)
        k_gain_norm = cfg.max_k_gain

    per_layer_ht = {}
    per_layer_delta = {}
    theta_new_flat = theta_current.squeeze()
    theta_old_flat = theta_current_in.squeeze()
    for L in range(info['num_nd_layers']):
        nd = info['nd_layers'][L]
        ltype = nd['type']
        lidx = nd['local_idx']
        label = f"{ltype[0].upper()}{lidx}"
        
        per_layer_ht[label] = per_layer[L]['ht_norm']
        
        w_s, w_l = nd['W_start'], nd['W_len']
        b_s, b_l = nd['b_start'], nd['b_len']
        w_delta = torch.norm(theta_new_flat[w_s:w_s+w_l] - theta_old_flat[w_s:w_s+w_l]).item()
        b_delta = torch.norm(theta_new_flat[b_s:b_s+b_l] - theta_old_flat[b_s:b_s+b_l]).item()
        per_layer_delta[label] = (w_delta**2 + b_delta**2)**0.5
        
    n_layers = info['num_nd_layers']
    gc = max(group_count, 1)
    debug_stats = {
        'innov_mean': total_innov_mean / gc,
        'innov_max': total_innov_max,
        'ht_norm': total_ht_norm / n_layers,
        'resid_norm': total_resid_norm / n_layers,
        'delta_y': total_delta_y / gc,
        'y_pred_norm': total_y_pred_norm / gc,
        'y_new': total_y_new / gc,
        'avg_P': total_avg_P / gc,
        'resid_in_innov': total_resid_in_innov / gc,
        'ht_theta_in_innov': total_ht_theta_in_innov / gc,
        'innov_norm': total_innov_norm / gc,
        'per_layer_ht': per_layer_ht,
        'per_layer_delta': per_layer_delta,
        # === NEW ===
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond_full,
    }
    
    return theta_current, new_S_info, (total_loss / layer_count).item(), target_var, k_gain_norm, debug_stats

# =========================================================================
# 10. Live Plotter (기존 6-subplot + 진단 플롯)
# =========================================================================
class LivePlotter:
    def __init__(self, method_name: str, max_episodes: int, param_str: str = ""):
        self.method_name = method_name
        self.outdir = cfg.outdir
        
        self.rewards, self.losses, self.p_inits, self.z_vars = [], [], [], []
        self.k_gains = [] 
        self.q_vals_0, self.q_vals_1 = [], []
        self.total_time, self.avg_step_time = 0.0, 0.0
        
        # === Diagnostic tracking ===
        self.cond_history = {}       # label -> [ep-avg cond over updates]
        self.ymax_history = {}
        self.theta_norm_history = {}
        self.null_ratio_history = []
        self.eff_rank_history = []
        self.stable_rank_history = []
        self.argmax_flip_history = []
        self.ref_dq_history = {name: [] for name in REF_NAMES}
        
        self.fig, self.axes = plt.subplots(1, 6, figsize=(30, 4))
        
        self.ax_r = self.axes[0]
        self.line_r_raw, = self.ax_r.plot([], [], 'b-', alpha=0.3)
        self.line_r_ma, = self.ax_r.plot([], [], 'b-', linewidth=2)
        self.ax_r.axhline(y=195, color='g', linestyle='--', alpha=0.5)
        self.ax_r.set_xlim(0, max_episodes); self.ax_r.set_ylim(0, 520)
        self.ax_r.set_title(f'Reward ({method_name})')
        
        self.ax_l = self.axes[1]
        self.line_l, = self.ax_l.plot([], [], 'r-', linewidth=1.5)
        self.ax_l.set_title('TD Loss'); self.ax_l.set_xlim(0, max_episodes)
        
        self.ax_p = self.axes[2]
        self.line_p, = self.ax_p.plot([], [], 'g-', linewidth=2)
        self.ax_p.set_title('P_init (Fixed)')
        self.ax_p.set_xlim(0, max_episodes)
        self.ax_p.set_ylim(0, cfg.p_init * 1.5)
            
        self.ax_z = self.axes[3]
        self.line_z, = self.ax_z.plot([], [], 'm-', linewidth=1.5)
        self.ax_z.set_title('TD Target Variance (Z_var)')
        self.ax_z.set_xlim(0, max_episodes)

        self.ax_k = self.axes[4]
        self.line_k, = self.ax_k.plot([], [], 'darkorange', linewidth=1.5)
        self.ax_k.set_title('Weight Update Norm ||Δθ||')
        self.ax_k.set_xlim(0, max_episodes)

        self.ax_q = self.axes[5]
        self.line_q0, = self.ax_q.plot([], [], 'c-', linewidth=1.5, label='Q(a=0) Left')
        self.line_q1, = self.ax_q.plot([], [], 'm-', linewidth=1.5, label='Q(a=1) Right')
        self.ax_q.set_title('Avg Q-Values')
        self.ax_q.set_xlim(0, max_episodes)
        self.ax_q.legend(loc='upper left')
        
        plt.tight_layout()
        
        prefix = f"{param_str}_" if param_str else ""
        clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        self.filename = os.path.join(self.outdir, f"{prefix}{clean_name}")
    
    def add(self, reward, loss, p_init=0.0, z_var=0.0, k_gain=0.0, q0=0.0, q1=0.0): 
        self.rewards.append(reward)
        self.losses.append(max(loss, 1e-10))
        self.p_inits.append(p_init)
        self.z_vars.append(z_var)
        self.k_gains.append(k_gain) 
        self.q_vals_0.append(q0)
        self.q_vals_1.append(q1)
    
    def add_diagnostics(self, cond_dict, ymax_dict, theta_norms, null_ratio,
                        eff_rank, stable_rank, argmax_flip, ref_q):
        if cond_dict:
            for k, v in cond_dict.items():
                self.cond_history.setdefault(k, []).append(v)
        if ymax_dict:
            for k, v in ymax_dict.items():
                self.ymax_history.setdefault(k, []).append(v)
        if theta_norms:
            for k, v in theta_norms.items():
                self.theta_norm_history.setdefault(k, []).append(v)
        self.null_ratio_history.append(null_ratio)
        self.eff_rank_history.append(eff_rank)
        self.stable_rank_history.append(stable_rank)
        self.argmax_flip_history.append(argmax_flip)
        if ref_q:
            for name in REF_NAMES:
                self.ref_dq_history[name].append(ref_q[name]['dq'])
    
    def refresh(self):
        ep_range = range(len(self.rewards))
        self.line_r_raw.set_data(ep_range, self.rewards)
        if len(self.rewards) >= 20:
            ma = np.convolve(self.rewards, np.ones(20)/20, 'valid')
            self.line_r_ma.set_data(range(19, len(self.rewards)), ma)
        
        self.line_l.set_data(ep_range, self.losses)
        self.line_p.set_data(ep_range, self.p_inits)
        self.line_z.set_data(ep_range, self.z_vars) 
        self.line_k.set_data(ep_range, self.k_gains) 
        
        self.line_q0.set_data(ep_range, self.q_vals_0)
        self.line_q1.set_data(ep_range, self.q_vals_1)
        
        for ax in self.axes:
            ax.relim(); ax.autoscale_view()
        self.axes[0].set_ylim(0, max(max(self.rewards, default=520) * 1.1, 520))
        
        plt.savefig(f'{self.filename}_live.png', dpi=100)
    
    def save_diagnostic_plots(self):
        """종료 후 진단 플롯 저장"""
        if not self.cond_history and not self.theta_norm_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
        
        # (0,0) Pseudo condition number per layer
        ax = axes[0, 0]
        for label, vals in sorted(self.cond_history.items()):
            ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log')
        ax.set_title('Pseudo Condition Number per Layer')
        ax.set_xlabel('Episode')
        ax.set_ylabel('cond(Y) approx')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # (0,1) Y_max per layer
        ax = axes[0, 1]
        for label, vals in sorted(self.ymax_history.items()):
            ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log')
        ax.set_title('Y_max per Layer (max eigenvalue approx)')
        ax.set_xlabel('Episode')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # (0,2) Layer theta norm
        ax = axes[0, 2]
        for label, vals in sorted(self.theta_norm_history.items()):
            ax.plot(vals, label=label, linewidth=1.5)
        ax.set_title('Layer ||θ|| Evolution')
        ax.set_xlabel('Episode')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # (1,0) Null ratio
        ax = axes[1, 0]
        ax.plot(self.null_ratio_history, 'r-', linewidth=2)
        ax.set_title('Advantage Null/Signal Ratio\n(Dueling identifiability; higher=worse)')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        # (1,1) Effective / Stable rank
        ax = axes[1, 1]
        ax.plot(self.eff_rank_history, 'b-', linewidth=2, label='effective rank')
        ax.plot(self.stable_rank_history, 'g--', linewidth=2, label='stable rank')
        ax.set_title('Shared Output Rank\n(low=feature collapse)')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (1,2) Reference states ΔQ
        ax = axes[1, 2]
        for name in REF_NAMES:
            ax.plot(self.ref_dq_history[name], label=name, linewidth=1.5)
        ax.set_title('ΔQ = Q(right) - Q(left) at Reference States')
        ax.set_xlabel('Episode')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.filename}_diagnostics.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[*] 진단 플롯 저장: {self.filename}_diagnostics.png")
        
        # Argmax flip 별도 플롯
        if self.argmax_flip_history:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(self.argmax_flip_history, 'orange', linewidth=1.5)
            ax2.set_title('Argmax Flip Rate (update-induced policy instability)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Flip rate')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.tight_layout()
            plt.savefig(f'{self.filename}_argmax_flip.png', dpi=120)
            plt.close(fig2)
    
    def close(self):
        plt.close(self.fig)

# =========================================================================
# 11. Landscape Visualization (unchanged)
# =========================================================================
def plot_cartpole_state_landscape(theta_star, info, cfg, normalizer, method_name, param_str, resolution=50):
    print(f"\n[Landscape] {method_name} 상태 공간 Q-지형 분석 중...")
    device = cfg.device
    theta_range = np.linspace(-0.25, 0.25, resolution)
    theta_dot_range = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(theta_range, theta_dot_range)
    states = np.zeros((resolution * resolution, 4))
    states[:, 2] = X.flatten(); states[:, 3] = Y.flatten()
    states_t = torch.tensor(states, dtype=torch.float64, device=device)
    if normalizer: states_t = normalizer.normalize(states_t)
        
    with torch.no_grad():
        q_vals_f32 = forward_single(theta_star.squeeze(), info, states_t.t())
        max_q = q_vals_f32.max(dim=0).values.cpu().numpy()
        
    Z = max_q.reshape(resolution, resolution)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
    
    z_min, z_max = np.min(Z), np.max(Z)
    z_floor = z_min - (z_max - z_min) * 0.15 
    ax.contourf(X, Y, Z, zdir='z', offset=z_floor, cmap='plasma', alpha=0.5)
    ax.set_zlim(z_floor, z_max)
    ax.view_init(elev=25, azim=230)

    ax.set_title(f'State-Space Q-Landscape: {method_name}\n({param_str})')
    ax.set_xlabel('Pole Angle (rad)'); ax.set_ylabel('Angular Velocity (rad/s)'); ax.set_zlabel('Max Q-value')
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(cfg.outdir, f"{param_str}_{clean_name}_State_Land.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] 상태 공간 지형도 저장: {filename}")

# =========================================================================
# 12. Training Function (호라이즌별 진단 + 에피소드별 진단 통합)
# =========================================================================
def train_srrhuif_nd():
    set_all_seeds(cfg.seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    spas_str = "SPAS" if cfg.use_spas else "StdDQN"
    print(f"\n{'='*60}")
    print(f"Training SRRHUIF-ND ({spas_str}) v8.0 | Params: {info['total_params']} ")
    print(f"Settings: [R={cfg.r_std}, α={cfg.alpha}, β={cfg.beta}, P={cfg.p_init}]")
    print(f"Horizon: {cfg.N_horizon}, Batch: {cfg.batch_size}, Q_std: {cfg.q_std}, τ: {cfg.tau_srrhuif}")
    print(f"Diagnostics: horizon_cond={cfg.diag_horizon_cond}, "
          f"full_eig={cfg.use_full_eigvalsh}, ref_Q={cfg.diag_ref_states}, "
          f"argmax={cfg.diag_argmax_flip}, eff_rank={cfg.diag_eff_rank}")
    print(f"{'='*60}")

    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    nd_cache = NDCache(info, cfg, cfg.device)
    
    for n_per, grp in nd_cache.n_per_groups.items():
        print(f"  n_per={n_per}: layers {grp['layers']}, neurons={grp['total_neurons']}")
    
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size,
          'normalizer': normalizer, 'device': cfg.device}

    theta = initialize_theta(info, cfg.device).view(-1, 1)
    theta_target = theta.clone()
    neuron_S_info = [1e-6 * nd_cache.get(L)['eye_n_per'].unsqueeze(-1).expand(-1, -1, nd['fan_out']).clone() 
                     for L, nd in enumerate(info['nd_layers'])]

    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)
    batch_hist = deque(maxlen=cfg.N_horizon)
    
    logger = LivePlotter(f"SRRHUIF-ND ({spas_str})", cfg.max_episodes, cfg.param_str)
    steps_done = 0
    train_start_time = time.time()
    update_times = []
    
    theta_history = [] 
    theta_snapshots = {}
    
    prev_ep_delta = None

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_r, ep_l, ep_var, ep_k_gain, ep_start = 0, [], [], [], time.time()
        ep_q0, ep_q1 = [], []
        ep_i_mean, ep_i_max = [], []
        
        last_h_k_traj = []
        last_h_p_traj = []
        last_h_ht_traj = []
        last_h_resid_traj = []
        last_h_innov_decomp = []
        last_h_cos_traj = []
        last_h_layer_ht = []
        last_h_layer_delta = []
        last_h_layer_cond = []   # [h_idx][label] -> cond
        last_h_layer_ymax = []   # [h_idx][label] -> ymax
        last_ep_cos = None
        
        # episode-level cond/ymax aggregation (averaged over all updates in ep)
        ep_cond_collect = {}   # label -> [cond values from h=last, per update]
        ep_ymax_collect = {}
        ep_argmax_flips = []
        
        theta_ep_start = theta.squeeze().clone()
        
        p_init = cfg.p_init

        for t in range(cfg.max_steps):
            steps_done += 1
            eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-steps_done / cfg.eps_decay_steps)
            
            with torch.no_grad():
                s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                s_t = s_t_buffer
                if normalizer: s_t = normalizer.normalize(s_t)
                q_vals = forward_single(theta.squeeze(), info, s_t).squeeze()
                ep_q0.append(q_vals[0].item())
                ep_q1.append(q_vals[1].item())

            if np.random.rand() < eps: 
                a = env.action_space.sample()
            else:
                a = q_vals.argmax().item()

            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)
                batch_hist.append(batch)
                if len(batch_hist) == cfg.N_horizon:
                    h_k_traj = []
                    h_p_traj = []
                    h_ht_traj = []
                    h_resid_traj = []
                    h_resid_in_innov_traj = []
                    h_ht_theta_traj = []
                    h_innov_traj = []
                    h_cos_traj = []
                    h_layer_ht = []
                    h_layer_delta = []
                    h_layer_cond = []    # NEW
                    h_layer_ymax = []    # NEW
                    prev_h_delta = None
                    q_next_caches = []
                    with torch.no_grad():
                        for h in range(cfg.N_horizon):
                            s_next_h = batch_hist[h]['s_next'].t()
                            if normalizer:
                                s_next_h = normalizer.normalize(s_next_h)
                            Q_tgt_h = forward_single(theta_target.squeeze(), info, s_next_h)
                            q_next_caches.append(Q_tgt_h)

                    # === Argmax flip tracking: capture state before horizon
                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            s_flip = batch_hist[0]['s'].t()
                            if normalizer:
                                s_flip = normalizer.normalize(s_flip)
                            Q_before = forward_single(theta.squeeze(), info, s_flip)
                            argmax_before = Q_before.argmax(dim=0)

                    for h in range(cfg.N_horizon):
                        is_first = (h == 0)
                        theta_before_h = theta.squeeze().clone()
                        
                        theta, neuron_S_info, l_val, t_var, t_k_gain, dbg = srrhuif_step_nd(
                            theta, theta_target, neuron_S_info, batch_hist[h], sp,
                            is_first, p_init, nd_cache,
                            q_next_target_cached=q_next_caches[h])
                        
                        h_delta = theta.squeeze() - theta_before_h
                        if prev_h_delta is not None:
                            d_norm = torch.norm(h_delta)
                            p_norm = torch.norm(prev_h_delta)
                            if d_norm > 1e-10 and p_norm > 1e-10:
                                cos = F.cosine_similarity(h_delta.unsqueeze(0), prev_h_delta.unsqueeze(0)).item()
                            else:
                                cos = 0.0
                            h_cos_traj.append(cos)
                        prev_h_delta = h_delta.clone()
                        
                        ep_l.append(l_val); ep_var.append(t_var); ep_k_gain.append(t_k_gain)
                        ep_i_mean.append(dbg['innov_mean']); ep_i_max.append(dbg['innov_max'])
                        h_k_traj.append(t_k_gain)
                        h_p_traj.append(dbg['avg_P'])
                        h_ht_traj.append(dbg['ht_norm'])
                        h_resid_traj.append(dbg['resid_norm'])
                        h_resid_in_innov_traj.append(dbg['resid_in_innov'])
                        h_ht_theta_traj.append(dbg['ht_theta_in_innov'])
                        h_innov_traj.append(dbg['innov_norm'])
                        h_layer_ht.append(dbg['per_layer_ht'])
                        h_layer_delta.append(dbg['per_layer_delta'])
                        # NEW: per-layer cond/ymax per horizon step
                        h_layer_cond.append(dbg['per_layer_cond'])
                        h_layer_ymax.append(dbg['per_layer_ymax'])

                    # === Argmax flip after horizon
                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            Q_after = forward_single(theta.squeeze(), info, s_flip)
                            argmax_after = Q_after.argmax(dim=0)
                            flip = (argmax_before != argmax_after).float().mean().item()
                            ep_argmax_flips.append(flip)
                    
                    # Aggregate cond/ymax over this update (use last horizon step)
                    if h_layer_cond:
                        last_cond_this_update = h_layer_cond[-1]
                        last_ymax_this_update = h_layer_ymax[-1]
                        for k, v in last_cond_this_update.items():
                            ep_cond_collect.setdefault(k, []).append(v)
                        for k, v in last_ymax_this_update.items():
                            ep_ymax_collect.setdefault(k, []).append(v)

                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                    last_h_k_traj = h_k_traj
                    last_h_p_traj = h_p_traj
                    last_h_ht_traj = h_ht_traj
                    last_h_resid_traj = h_resid_traj
                    last_h_innov_decomp = list(zip(h_resid_in_innov_traj, h_ht_theta_traj, h_innov_traj))
                    last_h_cos_traj = h_cos_traj
                    last_h_layer_ht = h_layer_ht
                    last_h_layer_delta = h_layer_delta
                    last_h_layer_cond = h_layer_cond   # NEW
                    last_h_layer_ymax = h_layer_ymax   # NEW
                    
                update_times.append(time.perf_counter() - update_start) 
            if done or trunc: break

        avg_l = np.mean(ep_l) if ep_l else 0
        avg_v = np.mean(ep_var) if ep_var else 0 
        avg_k = np.mean(ep_k_gain) if ep_k_gain else 0 
        avg_q0 = np.mean(ep_q0) if ep_q0 else 0
        avg_q1 = np.mean(ep_q1) if ep_q1 else 0
        avg_i_mean = np.mean(ep_i_mean) if ep_i_mean else 0
        max_i_max = np.max(ep_i_max) if ep_i_max else 0
        
        logger.add(ep_r, avg_l, p_init, avg_v, avg_k, avg_q0, avg_q1)
        
        # === EPISODE-LEVEL DIAGNOSTICS ===
        theta_norms = compute_layer_theta_norms(theta, info)
        null_ratio, null_abs, signal_abs = compute_advantage_null_ratio(theta, info)
        
        eff_rank_val, stable_rank_val = -1.0, -1.0
        if cfg.diag_eff_rank and buffer.current_size >= 128:
            with torch.no_grad():
                diag_batch = buffer.sample_batch(min(256, buffer.current_size))
                s_diag = diag_batch['s'].t()
                if normalizer:
                    s_diag = normalizer.normalize(s_diag)
                _, shared_out = forward_single_with_shared(theta.squeeze(), info, s_diag)
                eff_rank_val, stable_rank_val = compute_effective_rank(shared_out)
        
        avg_cond_dict = {k: float(np.mean(v)) for k, v in ep_cond_collect.items()}
        avg_ymax_dict = {k: float(np.mean(v)) for k, v in ep_ymax_collect.items()}
        avg_argmax_flip = float(np.mean(ep_argmax_flips)) if ep_argmax_flips else 0.0
        
        ref_q = None
        if cfg.diag_ref_states:
            ref_q = compute_ref_q_values(theta, info, normalizer, cfg.device)
        
        logger.add_diagnostics(avg_cond_dict, avg_ymax_dict, theta_norms,
                               null_ratio, eff_rank_val, stable_rank_val,
                               avg_argmax_flip, ref_q)
        
        ep_delta = theta.squeeze() - theta_ep_start
        ep_delta_norm = torch.norm(ep_delta).item()
        if prev_ep_delta is not None and ep_delta_norm > 1e-10 and torch.norm(prev_ep_delta) > 1e-10:
            last_ep_cos = F.cosine_similarity(ep_delta.unsqueeze(0), prev_ep_delta.unsqueeze(0)).item()
        else:
            last_ep_cos = None
        prev_ep_delta = ep_delta.clone()
        
        target_drift = torch.norm(theta_target.squeeze() - theta.squeeze()).item()
        
        theta_history.append(theta.squeeze().clone().cpu().numpy())
        
        if ep % 20 == 0 or ep == cfg.max_episodes:
            theta_snapshots[ep] = theta.clone()

        if ep % cfg.plot_interval == 0: logger.refresh()
        if ep % cfg.log_interval == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            print(f"[SRRHUIF] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} | eps: {eps:.2f} | Buf: {buffer.current_size}/{cfg.buffer_size} "
                  f"| Loss: {avg_l:.4f} | T_Var: {avg_v:.4f} | P: {p_init:.4f} | K_Gain: {avg_k:.4f} "
                  f"| Q(0): {avg_q0:.2f} | Q(1): {avg_q1:.2f} | Time: {time.time()-ep_start:.2f}s")

            print(f"          └─▶ Innov (Mean / Max): [{avg_i_mean:.4f} / {max_i_max:.4f}]")
            if last_h_k_traj:
                fmt = lambda traj: "[" + ", ".join([f"{v:.4f}" for v in traj]) + "]"
                fmt_e = lambda traj: "[" + ", ".join([f"{v:.2e}" for v in traj]) + "]"
                fmt2 = lambda traj: "[" + ", ".join([f"{v:+.3f}" for v in traj]) + "]"
                print(f"          └─▶ K_Gain/h:  {fmt(last_h_k_traj)}")
                print(f"          └─▶ P_avg/h:   {fmt_e(last_h_p_traj)}")
                if last_h_innov_decomp:
                    r_parts = [d[0] for d in last_h_innov_decomp]
                    h_parts = [d[1] for d in last_h_innov_decomp]
                    i_parts = [d[2] for d in last_h_innov_decomp]
                    print(f"          └─▶ |z-ẑ|/h:   {fmt(r_parts)}")
                    print(f"          └─▶ |H^Tθ|/h:  {fmt(h_parts)}")
                    print(f"          └─▶ |innov|/h:  {fmt(i_parts)}")
                if last_h_cos_traj:
                    print(f"          └─▶ cos(δ)/h:  {fmt2(last_h_cos_traj)}   ← 방향 일관성")
                ep_cos_str = f"{last_ep_cos:+.3f}" if last_ep_cos is not None else "N/A"
                print(f"          └─▶ ep_cos: {ep_cos_str} | θ-target drift: {target_drift:.4f} | ep_Δθ: {ep_delta_norm:.4f}")
                
                if last_h_layer_ht:
                    labels = sorted(last_h_layer_ht[0].keys())
                    
                    for h_idx in range(len(last_h_layer_ht)):
                        ht_str = " ".join([f"{l}={last_h_layer_ht[h_idx][l]:.2f}" for l in labels])
                        print(f"          └─▶ ||H^T|| h={h_idx}:  {ht_str}")
                    
                    for h_idx in range(len(last_h_layer_delta)):
                        dk_str = " ".join([f"{l}={last_h_layer_delta[h_idx][l]:.4f}" for l in labels])
                        print(f"          └─▶ ||Δθ||  h={h_idx}:  {dk_str}")
                    
                    max_ht_per_layer = {l: max(last_h_layer_ht[h][l] for h in range(len(last_h_layer_ht))) for l in labels}
                    dominant = max(max_ht_per_layer, key=max_ht_per_layer.get)
                    print(f"          └─▶ Dominant layer: {dominant} (max||H^T||={max_ht_per_layer[dominant]:.1f})")
            
            # =====================================================================
            # === NEW DIAGNOSTIC OUTPUT ===
            # =====================================================================
            print(f"          ══ DIAGNOSTICS ══")
            
            # 호라이즌별 cond/Y_max
            if last_h_layer_cond:
                labels = sorted(last_h_layer_cond[0].keys())
                
                for h_idx in range(len(last_h_layer_cond)):
                    cond_str = " ".join([f"{l}={last_h_layer_cond[h_idx][l]:.1e}" for l in labels])
                    print(f"          ├─▶ cond(Y)h={h_idx}: {cond_str}")
                
                for h_idx in range(len(last_h_layer_ymax)):
                    ymax_str = " ".join([f"{l}={last_h_layer_ymax[h_idx][l]:.1e}" for l in labels])
                    print(f"          ├─▶ Y_max  h={h_idx}: {ymax_str}")
            
            # Layer theta norms
            labels = sorted(theta_norms.keys())
            norm_str = " ".join([f"{l}={theta_norms[l]:.3f}" for l in labels])
            print(f"          ├─▶ ||θ|| per layer:   {norm_str}")
            
            # Advantage null ratio
            print(f"          ├─▶ Adv null/signal:   ratio={null_ratio:.4f} (null={null_abs:.4f}, signal={signal_abs:.4f})")
            
            # Shared eff rank
            if eff_rank_val > 0:
                shared_dim = cfg.shared_layers[-1]
                print(f"          ├─▶ Shared rank:       eff={eff_rank_val:.1f}/{shared_dim}, stable={stable_rank_val:.2f}")
            
            # Argmax flip
            print(f"          ├─▶ Argmax flip rate:  {avg_argmax_flip:.4f} (updates={len(ep_argmax_flips)})")
            
            # Reference states
            if ref_q:
                ref_str = " ".join([f"{name}:ΔQ={ref_q[name]['dq']:+.4f}(a={ref_q[name]['argmax']})" for name in REF_NAMES])
                print(f"          └─▶ Ref states:        {ref_str}")

    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
    env.close()
    logger.refresh()
    logger.save_diagnostic_plots()
    
    try:
        plot_cartpole_state_landscape(theta, info, cfg, normalizer, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
    except Exception as e: print(f"[경고] 지형도 생성 중 오류 발생: {e}")
        
    logger.close()
    return logger

def main():
    print(f"\n{'#' * 70}")
    print(f"  SRRHUIF-ND v8.0 ({'SPAS' if cfg.use_spas else 'Standard'}) FullDiag Session")
    print(f"  Horizon: {cfg.N_horizon} | Batch: {cfg.batch_size} | Q_std: {cfg.q_std}")
    print(f"  Output Dir: {cfg.outdir} | Prefix: {cfg.param_str}")
    print(f"{'#' * 70}")

    srrhuif_log = train_srrhuif_nd()
    print("\n[✔] 실험 및 시각화가 완료되었습니다. (결과 폴더를 확인하세요!)")

if __name__ == "__main__":
    main()
