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

print("=" * 70)
print(f"SRRHUIF-D3QN v6.0 (Optimized FIR) | PyTorch: {torch.__version__}")
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
# 1. Configuration (FIR Philosophy: fixed params, no adaptive schedule)
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_episodes: int = 150
    max_steps: int = 500
    batch_size: int = 64        # ← 줄임 (32×16=512 총량 유지)
    buffer_size: int = 10000

    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    gamma: float = 0.94
    scale_factor: float = 1.0
    
    # --- SRRHUIF ND params (FIR: all fixed) ---
    tau_srrhuif: float = 0.02    # ← 올림 (긴 호라이즌 → 좋은 추정 → 빠른 추적)
    N_horizon: int = 9          # ← 늘림 (burn-in + estimation 확보)
    q_std: float = 5e-4        # ← 올림 (P가 호라이즌 끝까지 살아있게)
    r_std: float = 1.3         # ← 약간 올림

    alpha: float = 0.99
    beta: float = 2.0   
    kappa: float = 0.0
    
    max_k_gain: float = 0.0
    
    # Fixed P_init (acting as p_base for Layer-wise He-scaling)
    p_init: float = 0.08       # ★ 변경: 0.013 -> 0.08 (He-scale의 Base 값 역할)
    value_layer_scale : float = 0.2
    use_spas: bool = True  

    # --- Exploration ---
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000

    update_interval: int = 4
    use_input_norm: bool = True
    use_compile: bool = True
    plot_interval: int = 10
    seed: int = 0

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_std
        self.r_inv = 1.0 / (self.r_std ** 2)

        self.param_str = f"a{self.alpha}_b{self.beta}_r{self.r_std}_p{self.p_init}"
        self.outdir = f"./results_cartpole/{self.param_str}"
        os.makedirs(self.outdir, exist_ok=True)


cfg = Config()

# ── Argument Parsing ──
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
cfg.__post_init__()

# =========================================================================
# 2. Network Info & Cache (with n_per grouping)
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

        # ★ n_per 그룹별 사전 계산 (QR batching용)
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
        
        # 그룹별 사전 할당 텐서
        for n_per, grp in self.n_per_groups.items():
            total_n = grp['total_neurons']
            grp['eye_grouped'] = torch.eye(n_per, dtype=DTYPE, device=device).unsqueeze(0).expand(total_n, -1, -1).clone()
            grp['S_Q_grouped'] = cfg.q_std * grp['eye_grouped'].clone()
            grp['gamma'] = self.layers[grp['layers'][0]]['gamma']
            # fan_out 누적 오프셋 (split용)
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
# 6. ND Core Functions (unchanged math, called on grouped tensors)
# =========================================================================
def _nd_time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    combined = torch.cat([P_sqrt_prev, S_Q_cached], dim=2)
    P_sqrt_pred = tria_operation_batch(combined)               # ★ grouped → 1 QR call
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
    
    # ★ Debug stats
    ht_norm = torch.norm(HT_all, dim=1).mean().item()       # avg column norm across neurons & batch
    resid_norm = torch.norm(residual_all, dim=1).mean().item()
    
    return HT_all, residual_all, z_hat_f64, ht_norm, resid_norm

def _nd_meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all, r_inv_sqrt, r_inv, eye_batch):
    combined = torch.cat([S_pred, HT_all * r_inv_sqrt], dim=2)
    S_new_all = tria_operation_batch(combined)                  # ★ grouped → 1 QR call
    
    # ★ innov 분해: innov = residual + HT^T @ θ_prior
    ht_theta = torch.bmm(HT_all.transpose(1, 2), theta_3d)    # (neurons, batch, 1)
    innov = residual_all + ht_theta
    
    # ★ 각 항의 norm (batch 평균)
    innov_abs = torch.abs(innov)
    innov_mean = torch.mean(innov_abs)
    innov_max = torch.max(innov_abs)
    resid_in_innov = torch.mean(torch.abs(residual_all)).item()   # |z - ẑ| 기여
    ht_theta_in_innov = torch.mean(torch.abs(ht_theta)).item()    # |H^T θ| 기여
    innov_norm = innov_mean.item()                                 # |innov| 전체
    
    y_new_all = y_pred + torch.bmm(HT_all, r_inv * innov)
    
    # ★ Δy = HT @ R⁻¹ @ innov
    delta_y = torch.bmm(HT_all, r_inv * innov)
    delta_y_norm = torch.norm(delta_y, dim=1).mean()
    y_pred_norm = torch.norm(y_pred, dim=1).mean().item()
    y_new_norm = torch.norm(y_new_all, dim=1).mean()
    
    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    
    # ★ P_new approx from S_new diagonal
    S_diag = torch.diagonal(S_new_all, dim1=-2, dim2=-1)
    P_approx = 1.0 / (S_diag ** 2 + 1e-12)
    avg_P_new = P_approx.mean().item()
    
    meas_stats = {
        'innov_mean': innov_mean, 'innov_max': innov_max,
        'resid_in_innov': resid_in_innov,      # |z - ẑ| part
        'ht_theta_in_innov': ht_theta_in_innov, # |H^T θ| part
        'innov_norm': innov_norm,                # |innov| total
        'delta_y': delta_y_norm.item(),
        'y_pred_norm': y_pred_norm,
        'y_new_norm': y_new_norm.item(),
        'avg_P': avg_P_new,
    }
    return theta_new_all, S_new_all, meas_stats
# =========================================================================
# 7. Initialize theta & SRRHUIF ND Step (Optimized)
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
    Optimized SRRHUIF-ND step with:
      1. n_per grouped QR operations (time update & meas update)
      2. Target Q caching (q_next_target_cached)
      3. Layer-wise He-Scaled P_init (Spatial Normalization)
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info_dict = {}  # L -> S_info (will be sorted later)
    total_loss, layer_count = 0.0, 0

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    unified = nd_cache.unified_thetas
    unified[:] = theta_current.squeeze().to(DTYPE_FWD)

    # =====================================================================
    # Phase 1: Time Update — GROUPED by n_per
    # =====================================================================
    # 1a. Per-layer preparation (extract priors, P_sqrt_prev)
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

        # =====================================================================
        # ★ Layer-wise He-Scaled P_init 계산
        # =====================================================================
        layer_type = nd_layer['type']
        
        # 1. He-init 기반의 Layer-wise P_init 
        # (p_init_val을 p_base로 사용하여 fan_in 차원에 반비례하게 망치 크기 분배)
        current_p_init = p_init_val * (2.0 / max(1, fan_in))
        
        # 2. 마지막 출력층(V1, A1) 식별 및 P_init 스케일 억제 (거대한 Q-value 스케일 방어)
        is_output_layer = ('value' in layer_type or 'advantage' in layer_type) and fan_out <= 2
        if is_output_layer:
            current_p_init *= cfg.value_layer_scale  # 출력층은 cực도로 민감하므로 보폭을 강하게 누름
        # =====================================================================

        S_3d = neuron_S_info[L]
        if is_first or S_3d is None:
            # 계산된 current_p_init으로 각 레이어의 불확실성을 초기화
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

    # 1b. Grouped time update (one QR call per n_per group)
    for n_per_val, grp in nd_cache.n_per_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        # Concatenate along dim=0 (neuron batch dimension)
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev'] for L in layers_in_grp], dim=0)

        # ★ Single grouped call (replaces len(layers_in_grp) separate calls)
        S_pred_g, Y_pred_g, y_pred_g, X_sigma_g, scaled_P_g = _nd_time_update_core(
            all_theta_3d, all_P_sqrt, grp['S_Q_grouped'], grp['eye_grouped'], grp['gamma'])

        # Split results back to per-layer
        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            per_layer[L]['S_pred'] = S_pred_g[s:e]
            per_layer[L]['Y_pred'] = Y_pred_g[s:e]
            per_layer[L]['y_pred'] = y_pred_g[s:e]
            per_layer[L]['X_sigma_all'] = X_sigma_g[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]

    # 1c. Sigma point scatter (per-layer, different W_start positions — fast)
    for L in range(info['num_nd_layers']):
        pl = per_layer[L]
        lc = pl['lc']
        X_sigma_f32 = pl['X_sigma_all'].to(DTYPE_FWD)
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(pl['fan_out'], lc['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :pl['fan_in']])
        layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, pl['fan_in']:pl['fan_in'] + 1])

    # =====================================================================
    # Phase 2: z_measured computation (with Target Q caching)
    # =====================================================================
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

    # =====================================================================
    # Phase 3: H computation (per-layer, different fwd_slices — no QR, fast)
    # =====================================================================
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

    # =====================================================================
    # Phase 4: Measurement Update — GROUPED by n_per
    # =====================================================================

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
    }
    
    return theta_current, new_S_info, (total_loss / layer_count).item(), target_var, k_gain_norm, debug_stats
# =========================================================================
# 10. Live Plotter (6-Subplots)
# =========================================================================
class LivePlotter:
    def __init__(self, method_name: str, max_episodes: int, param_str: str = ""):
        self.method_name = method_name
        self.outdir = cfg.outdir
        
        self.rewards, self.losses, self.p_inits, self.z_vars = [], [], [], []
        self.k_gains = [] 
        self.q_vals_0, self.q_vals_1 = [], []
        self.total_time, self.avg_step_time = 0.0, 0.0
        
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
        
        # P_init (now fixed — shown as reference)
        self.ax_p = self.axes[2]
        self.line_p, = self.ax_p.plot([], [], 'g-', linewidth=2)
        self.ax_p.set_title('P_init (Fixed Base)')
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
    
    def close(self):
        plt.close(self.fig)

# =========================================================================
# 11. Landscape & Trajectory Visualization Functions
# =========================================================================
def plot_cartpole_state_landscape(theta_star, info, cfg, normalizer, method_name, param_str, resolution=50):
    print(f"\n[Landscape] {method_name} 상태 공간(State-Space) Q-지형 분석 중...")
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

def plot_srrhuif_loss_landscape(theta_star, info, buffer, cfg, method_name, param_str, resolution=21, span=0.5):
    print(f"[Landscape] {method_name} 가중치 공간(Weight-Space) Loss 지형 분석 중...")
    device = cfg.device
    theta_star = theta_star.squeeze().clone()
    n_params = len(theta_star)
    
    torch.manual_seed(42)
    d1 = torch.randn(n_params, dtype=DTYPE, device=device)
    d2 = torch.randn(n_params, dtype=DTYPE, device=device)
    d2 = d2 - (torch.dot(d1, d2) / torch.dot(d1, d1)) * d1 
    d1 = d1 / torch.norm(d1); d2 = d2 / torch.norm(d2)
    scale = torch.norm(theta_star) * 0.1
    d1, d2 = d1 * scale, d2 * scale
    
    eval_batch_size = min(1000, buffer.current_size)
    batch = buffer.sample_batch(eval_batch_size)
    s_b = batch['s'].to(DTYPE_FWD); s_next_b = batch['s_next'].to(DTYPE_FWD)
    
    x_coords = np.linspace(-span, span, resolution)
    y_coords = np.linspace(-span, span, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z_loss = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            theta_new = theta_star + X[i, j] * d1 + Y[i, j] * d2
            with torch.no_grad():
                Q_vals_f32 = forward_single(theta_new, info, s_b)
                q_vals = Q_vals_f32[batch['a'], torch.arange(eval_batch_size, device=device)]
                Q_next_f32 = forward_single(theta_new, info, s_next_b)
                a_best = Q_next_f32.argmax(dim=0)
                q_next = Q_next_f32[a_best, torch.arange(eval_batch_size, device=device)]
                q_target = batch['r'] + cfg.gamma * (1 - batch['term']) * q_next.to(DTYPE)
                loss = F.mse_loss(q_vals.to(DTYPE), q_target)
                Z_loss[i, j] = loss.item()
                
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_loss, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.scatter(0, 0, Z_loss[resolution//2, resolution//2], color='red', s=100, label='Current $\\theta^*$')
    spread = cfg.alpha * np.sqrt(cfg.p_init)
    cx = spread * np.cos(np.linspace(0, 2*np.pi, 50)); cy = spread * np.sin(np.linspace(0, 2*np.pi, 50))
    ax.plot(cx, cy, np.min(Z_loss), color='orange', linestyle='--', linewidth=2, label=f'Spread (r={spread:.2f})')

    ax.set_title(f'Weight-Space Loss Landscape: {method_name}\n({param_str})')
    ax.set_xlabel('Direction 1 ($d_1$)'); ax.set_ylabel('Direction 2 ($d_2$)'); ax.set_zlabel('TD Loss')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.legend()
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(cfg.outdir, f"{param_str}_{clean_name}_Weight_Land.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[*] 가중치 공간 지형도 저장: {filename}")

def plot_srrhuif_2d_trajectory(theta_star, theta_history, p_inits, info, buffer, cfg, method_name, param_str, resolution=30, span=0.6):
    print(f"[Landscape] {method_name} 2D 가중치 궤적 지도(Trajectory Map) 분석 중...")
    device = cfg.device
    theta_star = theta_star.squeeze().clone()
    n_params = len(theta_star)
    
    torch.manual_seed(42)
    d1 = torch.randn(n_params, dtype=DTYPE, device=device)
    d2 = torch.randn(n_params, dtype=DTYPE, device=device)
    d2 = d2 - (torch.dot(d1, d2) / torch.dot(d1, d1)) * d1 
    d1 = d1 / torch.norm(d1); d2 = d2 / torch.norm(d2)
    scale = torch.norm(theta_star) * 0.1
    d1, d2 = d1 * scale, d2 * scale
    
    eval_batch_size = min(1000, buffer.current_size)
    batch = buffer.sample_batch(eval_batch_size)
    s_b = batch['s'].to(DTYPE_FWD); s_next_b = batch['s_next'].to(DTYPE_FWD)
    
    x_coords = np.linspace(-span, span, resolution)
    y_coords = np.linspace(-span, span, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z_loss = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            theta_new = theta_star + X[i, j] * d1 + Y[i, j] * d2
            with torch.no_grad():
                Q_vals_f32 = forward_single(theta_new, info, s_b)
                q_vals = Q_vals_f32[batch['a'], torch.arange(eval_batch_size, device=device)]
                Q_next_f32 = forward_single(theta_new, info, s_next_b)
                a_best = Q_next_f32.argmax(dim=0)
                q_next = Q_next_f32[a_best, torch.arange(eval_batch_size, device=device)]
                q_target = batch['r'] + cfg.gamma * (1 - batch['term']) * q_next.to(DTYPE)
                loss = F.mse_loss(q_vals.to(DTYPE), q_target)
                Z_loss[i, j] = loss.item()
                
    traj_x, traj_y = [], []
    d1_np, d2_np = d1.cpu().numpy(), d2.cpu().numpy()
    theta_star_np = theta_star.cpu().numpy()
    
    for th in theta_history:
        diff = th - theta_star_np
        x_proj = np.dot(diff, d1_np) / np.dot(d1_np, d1_np)
        y_proj = np.dot(diff, d2_np) / np.dot(d2_np, d2_np)
        traj_x.append(x_proj)
        traj_y.append(y_proj)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z_loss, levels=30, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax, label='TD Loss')
    
    for i in range(len(traj_x)-1):
        ax.plot(traj_x[i:i+2], traj_y[i:i+2], color=plt.cm.autumn(i/len(traj_x)), linewidth=2, alpha=0.8)
    
    ax.scatter(traj_x[0], traj_y[0], color='black', s=80, marker='x', label='Start (Ep 1)')
    ax.scatter(0, 0, color='red', s=120, marker='*', label='Optimal $\\theta^*$ (Ep Final)')
    
    if len(traj_x) >= 80:
        ep80_idx = min(79, len(traj_x)-1)
        ax.scatter(traj_x[ep80_idx], traj_y[ep80_idx], color='blue', s=80, label='Ep 80')
        spread_80 = cfg.alpha * np.sqrt(p_inits[ep80_idx])
        circle_80 = plt.Circle((traj_x[ep80_idx], traj_y[ep80_idx]), spread_80, color='blue', fill=False, linestyle='--', linewidth=1.5, label='Spread @ Ep 80')
        ax.add_patch(circle_80)
        
    ax.set_title(f'2D Optimization Trajectory Map: {method_name}\n({param_str})')
    ax.set_xlabel('Direction 1 ($d_1$) Projection'); ax.set_ylabel('Direction 2 ($d_2$) Projection')
    ax.legend(loc='upper right')
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(cfg.outdir, f"{param_str}_{clean_name}_Trajectory_2D.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] 2D 궤적 지도 저장: {filename}")

def plot_q_landscape_timelapse(theta_snapshots, info, cfg, normalizer, method_name, param_str, resolution=40):
    print(f"\n[Landscape] {method_name} 에피소드별 Q-지형 타임랩스 생성 중...")
    device = cfg.device
    theta_range = np.linspace(-0.25, 0.25, resolution)
    theta_dot_range = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(theta_range, theta_dot_range)
    states = np.zeros((resolution * resolution, 4))
    states[:, 2] = X.flatten(); states[:, 3] = Y.flatten()
    states_t = torch.tensor(states, dtype=torch.float64, device=device)
    if normalizer: states_t = normalizer.normalize(states_t)
        
    epochs = sorted(list(theta_snapshots.keys()))
    if not epochs: return
    
    n_plots = min(len(epochs), 6)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    global_vmin, global_vmax = 0, 20
    
    for i, ep in enumerate(epochs[:n_plots]):
        ax = axes[i]
        theta_ep = theta_snapshots[ep]
        with torch.no_grad():
            q_vals_f32 = forward_single(theta_ep.squeeze(), info, states_t.t())
            max_q = q_vals_f32.max(dim=0).values.cpu().numpy()
            
        Z = max_q.reshape(resolution, resolution)
        Z = np.nan_to_num(Z, nan=-1.0, posinf=50.0, neginf=-1.0)
        
        contour = ax.contourf(X, Y, Z, levels=20, cmap='plasma', vmin=global_vmin, vmax=global_vmax, extend='both')
        ax.set_title(f'Episode {ep}')
        ax.set_xlabel('Pole Angle')
        if i % cols == 0: ax.set_ylabel('Angular Velocity')
        
    fig.colorbar(contour, ax=axes.tolist(), label='Max Q-value', shrink=0.8, pad=0.02)
    fig.suptitle(f'Q-Landscape Evolution (Time-Lapse): {method_name}\n({param_str})', fontsize=16)
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(cfg.outdir, f"{param_str}_{clean_name}_Timelapse_Q.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] 타임랩스 지형도 저장: {filename}")

def plot_1d_loss_evolution(theta_snapshots, info, buffer, cfg, method_name, param_str, resolution=50, span=0.5):
    print(f"[Landscape] {method_name} 1D 손실 지형 진화도 생성 중...")
    device = cfg.device
    epochs = sorted(list(theta_snapshots.keys()))
    if not epochs: return
    
    final_theta = theta_snapshots[epochs[-1]].squeeze().clone()
    n_params = len(final_theta)
    torch.manual_seed(42)
    d1 = torch.randn(n_params, dtype=DTYPE, device=device)
    d1 = d1 / torch.norm(d1) * (torch.norm(final_theta) * 0.1)
    
    eval_batch_size = min(1000, buffer.current_size)
    batch = buffer.sample_batch(eval_batch_size)
    s_b = batch['s'].to(DTYPE_FWD); s_next_b = batch['s_next'].to(DTYPE_FWD)
    
    alphas = np.linspace(-span, span, resolution)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    
    for idx, ep in enumerate(epochs):
        theta_ep = theta_snapshots[ep].squeeze()
        losses = []
        for alpha in alphas:
            theta_new = theta_ep + alpha * d1
            with torch.no_grad():
                Q_vals_f32 = forward_single(theta_new, info, s_b)
                q_vals = Q_vals_f32[batch['a'], torch.arange(eval_batch_size, device=device)]
                Q_next_f32 = forward_single(theta_new, info, s_next_b)
                a_best = Q_next_f32.argmax(dim=0)
                q_next = Q_next_f32[a_best, torch.arange(eval_batch_size, device=device)]
                q_target = batch['r'] + cfg.gamma * (1 - batch['term']) * q_next.to(DTYPE)
                loss = F.mse_loss(q_vals.to(DTYPE), q_target)
                losses.append(loss.item())
                
        losses = np.nan_to_num(losses, nan=1e10, posinf=1e10)
        ax.plot(alphas, losses, label=f'Ep {ep}', color=colors[idx], linewidth=2, alpha=0.8)

    ax.set_yscale('log')
    ax.set_title(f'1D Loss Landscape Evolution: {method_name}\n({param_str})')
    ax.set_xlabel('Direction $d_1$ Distance from $\\theta_{ep}$')
    ax.set_ylabel('TD Loss (Log Scale)')
    ax.legend(title='Training Progress')
    ax.grid(True, alpha=0.3)
    
    clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(cfg.outdir, f"{param_str}_{clean_name}_1D_Loss_Evol.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] 1D 손실 지형 진화도 저장: {filename}")

# =========================================================================
# 12. Training Function (Optimized: Target Q Cache + No Adaptive P)
# =========================================================================
def train_srrhuif_nd():
    set_all_seeds(cfg.seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    spas_str = "SPAS" if cfg.use_spas else "StdDQN"
    print(f"\n{'='*60}")
    print(f"Training SRRHUIF-ND ({spas_str}) v6.0 | Params: {info['total_params']} ")
    print(f"Settings: [R={cfg.r_std}, α={cfg.alpha}, β={cfg.beta}, P_base={cfg.p_init}]")
    print(f"Horizon: {cfg.N_horizon}, Batch: {cfg.batch_size}, Q_std: {cfg.q_std}, τ: {cfg.tau_srrhuif}")
    print(f"Optimizations: Grouped QR + Target Q Cache + Layer-wise He-Scaled P_init")
    print(f"{'='*60}")

    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    nd_cache = NDCache(info, cfg, cfg.device)
    
    # Print grouping info
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
    
    # ★ Episode-level direction tracking
    prev_ep_delta = None

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_r, ep_l, ep_var, ep_k_gain, ep_start = 0, [], [], [], time.time()
        ep_q0, ep_q1 = [], []
        ep_i_mean, ep_i_max = [], [] # ★ 혁신값 저장 리스트 추가
        
        last_h_k_traj = []
        last_h_p_traj = []
        last_h_ht_traj = []
        last_h_resid_traj = []
        last_h_innov_decomp = []
        last_h_cos_traj = []
        last_h_layer_ht = []
        last_h_layer_delta = []
        last_ep_cos = None
        
        theta_ep_start = theta.squeeze().clone()
        
        # ★ Fixed P_init (acting as p_base for Layer-wise He-scaling)
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
                    # ★ Target Q 캐싱: 호라이즌 루프 전에 모든 배치에 대해 사전 계산
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
                    prev_h_delta = None
                    q_next_caches = []
                    with torch.no_grad():
                        for h in range(cfg.N_horizon):
                            s_next_h = batch_hist[h]['s_next'].t()
                            if normalizer:
                                s_next_h = normalizer.normalize(s_next_h)
                            Q_tgt_h = forward_single(theta_target.squeeze(), info, s_next_h)
                            q_next_caches.append(Q_tgt_h)  # (nA, batch_sz)

                    for h in range(cfg.N_horizon):
                        is_first = (h == 0)
                        theta_before_h = theta.squeeze().clone()
                        
                        # ★ Layer-wise He-Scaled P_init 로직이 srrhuif_step_nd 내부에 통합되어 있음 (p_init은 p_base로 전달됨)
                        theta, neuron_S_info, l_val, t_var, t_k_gain, dbg = srrhuif_step_nd(
                            theta, theta_target, neuron_S_info, batch_hist[h], sp,
                            is_first, p_init, nd_cache,
                            q_next_target_cached=q_next_caches[h])
                        
                        # ★ Cosine similarity between consecutive horizon steps
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

                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                    last_h_k_traj = h_k_traj
                    last_h_p_traj = h_p_traj
                    last_h_ht_traj = h_ht_traj
                    last_h_resid_traj = h_resid_traj
                    last_h_innov_decomp = list(zip(h_resid_in_innov_traj, h_ht_theta_traj, h_innov_traj))
                    last_h_cos_traj = h_cos_traj
                    last_h_layer_ht = h_layer_ht
                    last_h_layer_delta = h_layer_delta
                    
                update_times.append(time.perf_counter() - update_start) 
            if done or trunc: break

        avg_l = np.mean(ep_l) if ep_l else 0
        avg_v = np.mean(ep_var) if ep_var else 0 
        avg_k = np.mean(ep_k_gain) if ep_k_gain else 0 
        avg_q0 = np.mean(ep_q0) if ep_q0 else 0
        avg_q1 = np.mean(ep_q1) if ep_q1 else 0
        # ★ 에피소드 평균 및 최대 혁신값
        avg_i_mean = np.mean(ep_i_mean) if ep_i_mean else 0
        max_i_max = np.max(ep_i_max) if ep_i_max else 0
        
        logger.add(ep_r, avg_l, p_init, avg_v, avg_k, avg_q0, avg_q1)
        
        # ★ Episode-level direction tracking
        ep_delta = theta.squeeze() - theta_ep_start
        ep_delta_norm = torch.norm(ep_delta).item()
        if prev_ep_delta is not None and ep_delta_norm > 1e-10 and torch.norm(prev_ep_delta) > 1e-10:
            last_ep_cos = F.cosine_similarity(ep_delta.unsqueeze(0), prev_ep_delta.unsqueeze(0)).item()
        else:
            last_ep_cos = None
        prev_ep_delta = ep_delta.clone()
        
        # ★ Theta-target drift: how far target moved from initial
        target_drift = torch.norm(theta_target.squeeze() - theta.squeeze()).item()
        
        theta_history.append(theta.squeeze().clone().cpu().numpy())
        
        if ep % 20 == 0 or ep == cfg.max_episodes:
            theta_snapshots[ep] = theta.clone()

        if ep % cfg.plot_interval == 0: logger.refresh()
        if ep % 5 == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            print(f"[SRRHUIF] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} | eps: {eps:.2f} "
                  f"| Loss: {avg_l:.4f} | T_Var: {avg_v:.4f} | P_base: {p_init:.4f} | K_Gain: {avg_k:.4f} "
                  f"| Q(0): {avg_q0:.2f} | Q(1): {avg_q1:.2f} | Time: {time.time()-ep_start:.2f}s")

            # ★ 터미널에 확장된 디버그 출력
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
                    print(f"          └─▶ cos(δ)/h:  {fmt2(last_h_cos_traj)}   ← 방향 일관성 (+1=같은방향, -1=반전)")
                ep_cos_str = f"{last_ep_cos:+.3f}" if last_ep_cos is not None else "N/A"
                print(f"          └─▶ ep_cos: {ep_cos_str} | θ-target drift: {target_drift:.4f} | ep_Δθ: {ep_delta_norm:.4f}")
                if last_h_layer_ht and len(last_h_layer_ht) >= 2:
                    labels = sorted(last_h_layer_ht[0].keys())
                    ht_h0 = " ".join([f"{l}={last_h_layer_ht[0][l]:.2f}" for l in labels])
                    ht_hN = " ".join([f"{l}={last_h_layer_ht[-1][l]:.2f}" for l in labels])
                    dk_h0 = " ".join([f"{l}={last_h_layer_delta[0][l]:.4f}" for l in labels])
                    dk_hN = " ".join([f"{l}={last_h_layer_delta[-1][l]:.4f}" for l in labels])
                    max_ht_per_layer = {l: max(last_h_layer_ht[h][l] for h in range(len(last_h_layer_ht))) for l in labels}
                    dominant = max(max_ht_per_layer, key=max_ht_per_layer.get)
                    print(f"          └─▶ ||H^T|| h=0:  {ht_h0}")
                    print(f"          └─▶ ||H^T|| h={len(last_h_layer_ht)-1}:  {ht_hN}")
                    print(f"          └─▶ ||Δθ||  h=0:  {dk_h0}")
                    print(f"          └─▶ ||Δθ||  h={len(last_h_layer_delta)-1}:  {dk_hN}")
                    print(f"          └─▶ Dominant layer: {dominant} (max||H^T||={max_ht_per_layer[dominant]:.1f})")

    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
    env.close()
    logger.refresh()
    
    try:
        plot_cartpole_state_landscape(theta, info, cfg, normalizer, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
        plot_srrhuif_loss_landscape(theta, info, buffer, cfg, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
        plot_srrhuif_2d_trajectory(theta, theta_history, logger.p_inits, info, buffer, cfg, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
        plot_q_landscape_timelapse(theta_snapshots, info, cfg, normalizer, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
        plot_1d_loss_evolution(theta_snapshots, info, buffer, cfg, f"SRRHUIF-ND ({spas_str})", cfg.param_str)
    except Exception as e: print(f"[경고] 지형도 생성 중 오류 발생: {e}")
        
    logger.close()
    return logger

def main():
    print(f"\n{'#' * 70}")
    print(f"  SRRHUIF-ND v6.0 ({'SPAS' if cfg.use_spas else 'Standard'}) Optimized FIR Session")
    print(f"  Horizon: {cfg.N_horizon} | Batch: {cfg.batch_size} | Q_std: {cfg.q_std}")
    print(f"  Output Dir: {cfg.outdir} | Prefix: {cfg.param_str}")
    print(f"{'#' * 70}")

    srrhuif_log = train_srrhuif_nd()
    print("\n[✔] 실험 및 시각화가 완료되었습니다. (결과 폴더를 확인하세요!)")

if __name__ == "__main__":
    main()