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
import imageio

matplotlib.use('Agg')

"""
=========================================================================
SRRHUIF-D3QN v2 + ★ [F] VectorEnv (A+B+C+D+F)
=========================================================================
  기존 (A~D 유지 — 변경 없음):
    [A] Compile-Safe Math   [B] NDCache Pre-compute
    [C] torch.compile       [D] Unified Forward

  ★ [F] VectorEnv (10개 CartPole 병렬):
    - SyncVectorEnv: 매 step 10개 transition 동시 수집
    - Buffer diversity: 10개 환경 경험이 항상 혼합
    - Action selection: forward_bmm으로 10개 state 한번에 처리
    - Episode 추적: 환경별 독립, done 시 개별 기록
    
  SRRHUIF 함수 (srrhuif_step_nd, _nd_*_core 등): ★ 완전 동일 ★
  변경 범위: Config(num_envs), TensorReplayBuffer(push_batch), train()
=========================================================================
"""

print("="*70)
print(f"SRRHUIF-D3QN v2+VecEnv | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
print("="*70)

def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
JITTER = 1e-6

# =========================================================================
# 1. Configuration
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_episodes: int = 1500
    max_steps: int = 500
    batch_size: int = 128      
    buffer_size: int = 200000
    
    # ★ [F] VectorEnv
    num_envs: int = 10          # 병렬 환경 수
    
    shared_layers: List[int] = field(default_factory=lambda: [16,16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])
    
    gamma: float = 0.9         
    
    scale_factor_fv: float = 1.0
    scale_factor_ld: float = 1.0
    scale_factor_nd: float = 1.0
    
    tau_fv: float = 0.005
    tau_ld: float = 0.005
    tau_nd: float = 0.005
    
    exploration: str = "epsilon_greedy"
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 25000
    exploration_scale: float = 3.0
    
    N_horizon: int = 7 
    q_std: float = 5e-3
    r_std_fv: float = 2.95
    r_std_ld: float = 3 
    r_std_nd: float = 1.95
    ld_global_denom: bool = False   
    nd_global_denom: bool = False   
    
    alpha: float = 0.99
    beta: float = 2.0
    kappa: float = 0.0
    
    p_init_min: float = 0.001
    p_init_max: float = 0.01
    adaptive_window: int = 15
    
    use_input_norm: bool = True
    use_compile: bool = False
    plot_interval: int = 10
    seed: int = 0
    
    render : str = "human" #human : 직접 // rgb_array : gif or mp4
    
    def __post_init__(self):
        self.r_inv_sqrt_fv = 1.0 / self.r_std_fv
        self.r_inv_fv      = 1.0 / (self.r_std_fv ** 2)
        self.r_inv_sqrt_ld = 1.0 / self.r_std_ld
        self.r_inv_ld      = 1.0 / (self.r_std_ld ** 2)
        self.r_inv_sqrt_nd = 1.0 / self.r_std_nd
        self.r_inv_nd      = 1.0 / (self.r_std_nd ** 2)

cfg = Config()

# =========================================================================
# 2. Network Info & Cache (변경 없음)
# =========================================================================
def create_network_info(dimS: int, nA: int, config: Config) -> Dict:
    info = {'dimS': dimS, 'nA': nA, 'layers': [], 'ld_layers': [], 'nd_layers': []}
    idx, ld_idx = 0, 0
    def add_layers(sizes, type_str):
        nonlocal idx, ld_idx
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            layer = {
                'type': type_str, 'layer_idx': i,
                'W_start': idx, 'W_len': fan_out * fan_in, 'W_shape': (fan_out, fan_in),
                'b_start': idx + fan_out * fan_in, 'b_len': fan_out,
                'fan_in': fan_in, 'fan_out': fan_out,
            }
            idx += fan_out * fan_in + fan_out
            info['layers'].append(layer)
            info['ld_layers'].append({'global_idx': ld_idx, 'type': type_str, 'local_idx': i,
                'n_params': fan_out * fan_in + fan_out, 'param_start': layer['W_start'], 
                'param_end': layer['b_start'] + layer['b_len']})
            info['nd_layers'].append({'global_idx': ld_idx, 'type': type_str, 'local_idx': i,
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
    info['num_ld_layers'] = len(info['ld_layers'])
    info['num_nd_layers'] = len(info['nd_layers'])
    return info

class NDCache:
    """[B]+[D] Pre-compute + Unified Forward Buffer"""
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
            lamb = cfg.alpha**2 * (n_per + cfg.kappa) - n_per
            gamma = float(np.sqrt(n_per + lamb))
            Wm = torch.zeros(2*n_per+1, dtype=DTYPE, device=device)
            Wc = torch.zeros(2*n_per+1, dtype=DTYPE, device=device)
            Wm[0] = lamb / (n_per + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha**2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (n_per + lamb)
            S_Q_cached = cfg.q_std * eye_n_per_batch.clone()
            zero_col = torch.zeros(fan_out, n_per, 1, dtype=DTYPE, device=device)
            Wm_col = Wm.view(1, -1, 1).expand(fan_out, -1, -1).clone()
            self.layers[L] = {
                'w_col_idx': w_col_idx, 'b_col_idx': b_col_idx,
                'eye_n_per': eye_n_per, 'eye_n_per_batch': eye_n_per_batch,
                'Wm': Wm, 'Wc': Wc, 'gamma': gamma,
                'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per, 'num_sigma': num_sigma,
                'S_Q_cached': S_Q_cached, 'zero_col': zero_col, 'Wm_col': Wm_col,
            }
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], dtype=DTYPE, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards
        print(f"  [D] Unified Forward Buffer: {total_forwards} forwards × {info['total_params']} params")
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
# 3. Forward Functions (변경 없음)
# =========================================================================
def forward_single(theta, info, x):
    if theta.dim() == 2: theta = theta.squeeze()
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start']+layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start']+layer['b_len']].view(-1, 1)
        h = F.relu(W @ h + b)
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start']+layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start']+layer['b_len']].view(-1, 1)
        z = W @ v + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start']+layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start']+layer['b_len']].view(-1, 1)
        z = W @ a + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    return v + (a - a.mean(dim=0, keepdim=True))

def forward_bmm(thetas, info, x):
    num_sigma = thetas.shape[0]
    x_expanded = x.t().unsqueeze(0).expand(num_sigma, -1, -1)
    h = x_expanded
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start']+layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start']+layer['b_len']].view(num_sigma, out_dim, 1)
        h = F.relu(torch.bmm(W, h) + b)
    shared_out = h
    v = shared_out
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start']+layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start']+layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, v) + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = shared_out
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start']+layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start']+layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, a) + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    return v + (a - a.mean(dim=1, keepdim=True))

def forward_3part_ld(theta_current, info, x, opt_layer_idx, theta_sigma_L):
    num_sigma = theta_sigma_L.shape[0]
    thetas = theta_current.squeeze().unsqueeze(0).expand(num_sigma, -1).clone()
    ld_layer = info['ld_layers'][opt_layer_idx]
    start, end = ld_layer['param_start'], ld_layer['param_end']
    thetas[:, start:end] = theta_sigma_L
    return forward_bmm(thetas, info, x)

# =========================================================================
# ★ [F] Replay Buffer — push_batch 추가
# =========================================================================
class TensorReplayBuffer:
    def __init__(self, capacity: int, dimS: int, device: str):
        self.capacity = capacity
        self.count = 0
        self.device = device
        self.S      = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A      = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R      = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term   = torch.zeros(capacity, dtype=DTYPE, device=device)
    
    def push(self, s, a, r, s_next, done):
        idx = self.count % self.capacity
        self.S[idx]      = torch.as_tensor(s, dtype=DTYPE, device=self.device)
        self.A[idx]      = a
        self.R[idx]      = r
        self.S_next[idx] = torch.as_tensor(s_next, dtype=DTYPE, device=self.device)
        self.term[idx]   = float(done)
        self.count += 1
    
    def push_batch(self, s_batch, a_batch, r_batch, s_next_batch, done_batch):
        """★ [F] N개 transition을 한번에 push"""
        n = s_batch.shape[0]
        start = self.count % self.capacity
        end = start + n
        
        if end <= self.capacity:
            # 한 번에 들어가는 경우
            self.S[start:end]      = s_batch
            self.A[start:end]      = a_batch
            self.R[start:end]      = r_batch
            self.S_next[start:end] = s_next_batch
            self.term[start:end]   = done_batch
        else:
            # 순환 버퍼 경계를 넘는 경우
            first = self.capacity - start
            self.S[start:]       = s_batch[:first]
            self.A[start:]       = a_batch[:first]
            self.R[start:]       = r_batch[:first]
            self.S_next[start:]  = s_next_batch[:first]
            self.term[start:]    = done_batch[:first]
            
            rest = n - first
            self.S[:rest]       = s_batch[first:]
            self.A[:rest]       = a_batch[first:]
            self.R[:rest]       = r_batch[first:]
            self.S_next[:rest]  = s_next_batch[first:]
            self.term[:rest]    = done_batch[first:]
        
        self.count += n
    
    @property
    def current_size(self):
        return min(self.count, self.capacity)
    
    def sample_batch(self, batch_size: int) -> Dict:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return {
            's':      self.S[indices].t(),
            'a':      self.A[indices],
            'r':      self.R[indices],
            's_next': self.S_next[indices].t(),
            'term':   self.term[indices],
        }

# =========================================================================
# 4~5. Math Utilities & Step Functions (★ 완전 동일 — 변경 없음)
# =========================================================================
def tria_operation(A):
    _, r = torch.linalg.qr(A.t())
    s = r.t(); d = torch.diag(s)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    s = s * signs; s.diagonal().clamp_(min=1e-6); return s

def tria_operation_batch(A):
    _, r = torch.linalg.qr(A.transpose(-2, -1).contiguous())
    s = r.transpose(-2, -1).contiguous()
    d = torch.diagonal(s, dim1=-2, dim2=-1)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    s = s * signs.unsqueeze(-1); s.diagonal(dim1=-2, dim2=-1).clamp_(min=1e-6); return s

def safe_inv_tril(L, n, device):
    eye = torch.eye(n, dtype=L.dtype, device=device)
    result = torch.linalg.solve_triangular(L + JITTER * eye, eye, upper=False)
    return torch.where(torch.isfinite(result), result, eye)

def safe_inv_tril_batch(L_batch, eye_batch):
    result = torch.linalg.solve_triangular(L_batch + JITTER * eye_batch, eye_batch, upper=False)
    return torch.where(torch.isfinite(result), result, eye_batch)

def robust_solve_spd(S_tril, y, n, device):
    eye = torch.eye(n, dtype=S_tril.dtype, device=device)
    S_safe = S_tril + JITTER * eye
    z = torch.linalg.solve_triangular(S_safe, y, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.t().contiguous(), z, upper=True)
    return torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))

def robust_solve_spd_batch(S_tril_batch, y_batch, eye_batch):
    S_safe = S_tril_batch + JITTER * eye_batch
    z = torch.linalg.solve_triangular(S_safe, y_batch, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.transpose(-2, -1).contiguous(), z, upper=True)
    return torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))

def _nd_time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    combined = torch.cat([P_sqrt_prev, S_Q_cached], dim=2)
    P_sqrt_pred = tria_operation_batch(combined)
    S_pred = safe_inv_tril_batch(P_sqrt_pred, eye_batch)
    Y_pred = torch.bmm(S_pred, S_pred.transpose(-2, -1))
    y_pred = torch.bmm(Y_pred, theta_3d)
    scaled_P = gamma_val * P_sqrt_pred
    theta_2d = theta_3d.squeeze(-1)
    X_sigma_all = torch.cat([theta_2d.unsqueeze(1),
        theta_2d.unsqueeze(1) + scaled_P.transpose(-2, -1),
        theta_2d.unsqueeze(1) - scaled_P.transpose(-2, -1)], dim=1)
    return S_pred, Y_pred, y_pred, X_sigma_all, scaled_P

def _nd_compute_ht_core(Z_sigma_T, Wm_col, z_measured_exp, Y_pred, scaled_P, Wc, zero_col, eye_batch):
    z_hat_all = torch.bmm(Z_sigma_T, Wm_col)
    residual_all = z_measured_exp - z_hat_all
    X_dev_all = torch.cat([zero_col, scaled_P, -scaled_P], dim=2)
    Z_dev = Z_sigma_T - z_hat_all
    P_xz_all = torch.bmm(X_dev_all * Wc.view(1, 1, -1), Z_dev.transpose(1, 2))
    HT_all = torch.bmm(Y_pred, P_xz_all)
    return HT_all, residual_all, z_hat_all

def _nd_meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all, r_inv_sqrt, r_inv, eye_batch):
    combined = torch.cat([S_pred, HT_all * r_inv_sqrt], dim=2)
    S_new_all = tria_operation_batch(combined)
    innov = residual_all + torch.bmm(HT_all.transpose(1, 2), theta_3d)
    y_new_all = y_pred + torch.bmm(HT_all, r_inv * innov)
    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    return theta_new_all, S_new_all

if cfg.use_compile and hasattr(torch, 'compile') and cfg.device == 'cuda':
    _cm = 'default'
    print(f"Applying torch.compile(mode='{_cm}')...")
    forward_single = torch.compile(forward_single, mode=_cm)
    forward_bmm = torch.compile(forward_bmm, mode=_cm)
    _nd_time_update_core = torch.compile(_nd_time_update_core, mode=_cm)
    _nd_compute_ht_core = torch.compile(_nd_compute_ht_core, mode=_cm)
    _nd_meas_update_core = torch.compile(_nd_meas_update_core, mode=_cm)
    print("Compilation Scheduled.")
else:
    print("torch.compile disabled or CPU mode.")

# Thompson Sampling (변경 없음)
def compute_S_inv_t(S, n, device):
    eye = torch.eye(n, dtype=S.dtype, device=device)
    S_inv = torch.linalg.solve_triangular(S + JITTER * eye, eye, upper=False)
    S_inv_t = S_inv.t(); return torch.where(torch.isfinite(S_inv_t), S_inv_t, eye)

def compute_S_inv_t_batch(S_batch, n, device):
    eye = torch.eye(n, dtype=S_batch.dtype, device=device).unsqueeze(0).expand(S_batch.shape[0], -1, -1)
    S_inv = torch.linalg.solve_triangular(S_batch + JITTER * eye, eye, upper=False)
    S_inv_t = S_inv.transpose(-2, -1); return torch.where(torch.isfinite(S_inv_t), S_inv_t, eye)

def ts_sample_theta_fv(theta, S_inv_t_cached, ts_temperature, exploration_scale, n_x, device):
    w = torch.randn(n_x, 1, dtype=DTYPE, device=device)
    return theta.squeeze() + (ts_temperature * exploration_scale * (S_inv_t_cached @ w)).squeeze()

def ts_sample_theta_ld(theta, layer_S_inv_t_cached, ts_temperature, exploration_scale, info, device):
    theta_explored = theta.squeeze().clone()
    for L in range(info['num_ld_layers']):
        ld = info['ld_layers'][L]
        w = torch.randn(ld['n_params'], 1, dtype=DTYPE, device=device)
        theta_explored[ld['param_start']:ld['param_end']] += (ts_temperature * exploration_scale * (layer_S_inv_t_cached[L] @ w)).squeeze()
    return theta_explored

def ts_sample_theta_nd(theta, neuron_S_inv_t_cached, ts_temperature, exploration_scale, info, nd_cache, device):
    theta_explored = theta.squeeze().clone()
    for L in range(info['num_nd_layers']):
        nd_layer = info['nd_layers'][L]; lc = nd_cache.get(L)
        w_batch = torch.randn(lc['fan_out'], lc['n_per'], 1, dtype=DTYPE, device=device)
        noise = ts_temperature * exploration_scale * torch.bmm(neuron_S_inv_t_cached[L], w_batch)
        theta_explored[nd_layer['W_start']:nd_layer['W_start']+nd_layer['W_len']] += noise[:, :lc['fan_in'], 0].reshape(-1)
        theta_explored[nd_layer['b_start']:nd_layer['b_start']+nd_layer['b_len']] += noise[:, lc['fan_in'], 0]
    return theta_explored

def initialize_theta(info, device):
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    for layer in info['layers']:
        fan_in, W_len = layer['W_shape'][1], layer['W_len']
        theta[layer['W_start']:layer['W_start']+W_len] = torch.randn(W_len, dtype=DTYPE, device=device) * np.sqrt(2.0/fan_in)
    return theta

def compute_ut_weights(n, alpha, beta, kappa, device):
    lamb = alpha**2 * (n + kappa) - n
    Wm = torch.zeros(2*n+1, dtype=DTYPE, device=device)
    Wc = torch.zeros(2*n+1, dtype=DTYPE, device=device)
    Wm[0] = lamb / (n + lamb); Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 0.5 / (n + lamb)
    return Wm, Wc, float(np.sqrt(n + lamb))

def generate_sigma_points(theta_prior, P_sqrt, gamma, n, device):
    theta_flat = theta_prior.squeeze(); scaled_P = gamma * P_sqrt
    X_sigma = torch.cat([theta_flat.unsqueeze(0), theta_flat.unsqueeze(0) + scaled_P.t(),
        theta_flat.unsqueeze(0) - scaled_P.t()], dim=0)
    return X_sigma, scaled_P

# FV step (변경 없음)
@torch.no_grad()
def srrhuif_step_full_vector(theta_current, theta_target, S_info, batch, sp, is_first, p_init_val):
    device, n_x, batch_sz, info = sp['device'], sp['n_x'], sp['batch_sz'], sp['info']
    theta_pred = theta_target if is_first else theta_current
    P_sqrt_prev = p_init_val * torch.eye(n_x, dtype=DTYPE, device=device) if is_first else safe_inv_tril(S_info, n_x, device)
    S_Q = cfg.q_std * torch.eye(n_x, dtype=DTYPE, device=device)
    P_sqrt_pred = tria_operation(torch.cat([P_sqrt_prev, S_Q], dim=1))
    S_pred = safe_inv_tril(P_sqrt_pred, n_x, device); Y_pred = S_pred @ S_pred.t(); y_pred = Y_pred @ theta_pred
    X_sigma, scaled_P = generate_sigma_points(theta_pred, P_sqrt_pred, sp['gamma_sigma'], n_x, device)
    s_batch = batch['s'].t()
    if sp.get('normalizer'): s_batch = sp['normalizer'].normalize(s_batch)
    Q_all = forward_bmm(X_sigma, info, s_batch)
    Z_sigma = Q_all[:, batch['a'], torch.arange(batch_sz, device=device)].t()
    z_hat = Z_sigma @ sp['Wm'].view(-1, 1)
    s_next = batch['s_next'].t()
    if sp.get('normalizer'): s_next = sp['normalizer'].normalize(s_next)
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both = forward_bmm(thetas_pair, info, s_next)
    a_best_next = Q_both[0].argmax(dim=0)
    q_val_next = Q_both[1][a_best_next, torch.arange(batch_sz, device=device)]
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    residual = z_measured - z_hat
    X_dev = torch.cat([torch.zeros(n_x, 1, dtype=DTYPE, device=device), scaled_P, -scaled_P], dim=1)
    HT = Y_pred @ ((X_dev * sp['Wc']) @ (Z_sigma - z_hat).t())
    S_new = tria_operation(torch.cat([S_pred, HT * cfg.r_inv_sqrt_fv], dim=1))
    y_new = y_pred + HT @ (cfg.r_inv_fv * (residual + HT.t() @ theta_pred))
    theta_new = robust_solve_spd(S_new, y_new, n_x, device)
    if not torch.isfinite(theta_new).all(): theta_new = theta_pred.clone()
    return theta_new, S_new, torch.mean(residual**2).item()

# LD step (변경 없음)
@torch.no_grad()
def srrhuif_step_layer_decoupled(theta_current_in, theta_target, layer_S_info, batch, sp, is_first, p_init_val):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone(); new_S_info, total_loss = [], 0.0
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'): s_batch = sp['normalizer'].normalize(s_batch); s_next = sp['normalizer'].normalize(s_next)
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both = forward_bmm(thetas_pair, info, s_next); a_best_next = Q_both[0].argmax(dim=0)
    q_val_next = Q_both[1][a_best_next, torch.arange(batch_sz, device=device)]
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    layer_data, P_zz_sum = [], 0.0
    for L in range(info['num_ld_layers']):
        ld_layer = info['ld_layers'][L]; n_L, start, end = ld_layer['n_params'], ld_layer['param_start'], ld_layer['param_end']
        theta_L_prior = theta_prior.squeeze()[start:end].view(-1, 1); S_L = layer_S_info[L]
        P_sqrt_prev = p_init_val * torch.eye(n_L, dtype=DTYPE, device=device) if is_first or S_L is None else safe_inv_tril(S_L, n_L, device)
        S_Q = cfg.q_std * torch.eye(n_L, dtype=DTYPE, device=device)
        P_sqrt_pred = tria_operation(torch.cat([P_sqrt_prev, S_Q], dim=1)); S_pred = safe_inv_tril(P_sqrt_pred, n_L, device)
        Y_pred = S_pred @ S_pred.t(); y_pred = Y_pred @ theta_L_prior
        Wm_L, Wc_L, gamma_L = compute_ut_weights(n_L, cfg.alpha, cfg.beta, cfg.kappa, device)
        X_sigma_L, scaled_P_L = generate_sigma_points(theta_L_prior, P_sqrt_pred, gamma_L, n_L, device)
        Q_all = forward_3part_ld(theta_current, info, s_batch, L, X_sigma_L)
        Z_sigma = Q_all[:, batch['a'], torch.arange(batch_sz, device=device)].t(); z_hat = Z_sigma @ Wm_L.view(-1, 1)
        residual = z_measured - z_hat
        X_dev = torch.cat([torch.zeros(n_L, 1, dtype=DTYPE, device=device), scaled_P_L, -scaled_P_L], dim=1)
        HT = Y_pred @ ((X_dev * Wc_L) @ (Z_sigma - z_hat).t()); Z_dev_out = Z_sigma - z_hat
        P_zz_L = (Wc_L.view(1, -1) * Z_dev_out ** 2).sum(dim=1).mean().item(); P_zz_sum += P_zz_L
        layer_data.append({'n_L': n_L, 'start': start, 'end': end, 'theta_L_prior': theta_L_prior,
            'S_pred': S_pred, 'y_pred': y_pred, 'HT': HT, 'residual': residual,
            'loss': torch.mean(residual**2).item(), 'P_zz_L': P_zz_L})
    for L, data in enumerate(layer_data):
        if cfg.ld_global_denom:
            R_eff_L = max(cfg.r_std_ld ** 2 + (P_zz_sum - data['P_zz_L']), 1e-10)
            r_inv_sqrt_L, r_inv_L = 1.0 / np.sqrt(R_eff_L), 1.0 / R_eff_L
        else: r_inv_sqrt_L, r_inv_L = cfg.r_inv_sqrt_ld, cfg.r_inv_ld
        S_new = tria_operation(torch.cat([data['S_pred'], data['HT'] * r_inv_sqrt_L], dim=1))
        y_new = data['y_pred'] + data['HT'] @ (r_inv_L * (data['residual'] + data['HT'].t() @ data['theta_L_prior']))
        theta_L_new = robust_solve_spd(S_new, y_new, data['n_L'], device)
        if not torch.isfinite(theta_L_new).all(): theta_L_new = data['theta_L_prior'].clone()
        theta_current.squeeze()[data['start']:data['end']] = theta_L_new.squeeze()
        new_S_info.append(S_new); total_loss += data['loss']
    return theta_current, new_S_info, total_loss / info['num_ld_layers']

# ND step (★ 완전 동일)
@torch.no_grad()
def srrhuif_step_nd(theta_current_in, theta_target, neuron_S_info, batch, sp, is_first, p_init_val, nd_cache):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info, total_loss, layer_count = [], 0.0, 0
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'): s_batch = sp['normalizer'].normalize(s_batch); s_next = sp['normalizer'].normalize(s_next)
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both = forward_bmm(thetas_pair, info, s_next); a_best_next = Q_both[0].argmax(dim=0)
    q_val_next = Q_both[1][a_best_next, torch.arange(batch_sz, device=device)]
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    unified = nd_cache.unified_thetas; unified[:] = theta_current.squeeze()
    time_data = []
    for L in range(info['num_nd_layers']):
        nd_layer = info['nd_layers'][L]; lc = nd_cache.get(L)
        fan_in, fan_out, n_per = lc['fan_in'], lc['fan_out'], lc['n_per']
        W_start, b_start = nd_layer['W_start'], nd_layer['b_start']
        W_prior = theta_prior.squeeze()[W_start:W_start+nd_layer['W_len']].view(fan_out, fan_in)
        b_prior = theta_prior.squeeze()[b_start:b_start+nd_layer['b_len']]
        theta_all_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1)
        theta_all_prior_3d = theta_all_prior.unsqueeze(-1)
        S_3d = neuron_S_info[L]
        if is_first or S_3d is None: P_sqrt_prev = np.sqrt(p_init_val) * lc['eye_n_per_batch'].clone()
        else: P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_n_per_batch'])
        S_pred, Y_pred, y_pred, X_sigma_all, scaled_P = _nd_time_update_core(
            theta_all_prior_3d, P_sqrt_prev, lc['S_Q_cached'], lc['eye_n_per_batch'], lc['gamma'])
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_all[:, :, :fan_in])
        layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_all[:, :, fan_in:fan_in+1])
        time_data.append({'nd_layer': nd_layer, 'lc': lc, 'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per,
            'W_start': W_start, 'b_start': b_start, 'theta_all_prior': theta_all_prior,
            'theta_all_prior_3d': theta_all_prior_3d, 'S_pred': S_pred, 'Y_pred': Y_pred,
            'y_pred': y_pred, 'scaled_P': scaled_P})
    Q_all_unified = forward_bmm(unified, info, s_batch)
    layer_data = []
    for L in range(info['num_nd_layers']):
        td = time_data[L]; lc = td['lc']; fan_out = td['fan_out']
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        Q_L = Q_all_unified[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], info['nA'], -1)
        Z_sigma_T = Q_L[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
        z_measured_exp = z_measured.unsqueeze(0).expand(fan_out, -1, -1)
        HT_all, residual_all, _ = _nd_compute_ht_core(Z_sigma_T, lc['Wm_col'], z_measured_exp,
            td['Y_pred'], td['scaled_P'], lc['Wc'], lc['zero_col'], lc['eye_n_per_batch'])
        layer_data.append({'nd_layer': td['nd_layer'], 'lc': lc, 'fan_in': td['fan_in'], 'fan_out': fan_out,
            'n_per': td['n_per'], 'W_start': td['W_start'], 'b_start': td['b_start'],
            'theta_all_prior': td['theta_all_prior'], 'theta_all_prior_3d': td['theta_all_prior_3d'],
            'S_pred': td['S_pred'], 'y_pred': td['y_pred'], 'HT_all': HT_all, 'residual_all': residual_all,
            'loss': torch.mean(residual_all**2).item()})
        layer_count += 1
    for L, data in enumerate(layer_data):
        theta_new_all, S_new_all = _nd_meas_update_core(data['S_pred'], data['y_pred'], data['HT_all'],
            data['theta_all_prior_3d'], data['residual_all'], cfg.r_inv_sqrt_nd, cfg.r_inv_nd, data['lc']['eye_n_per_batch'])
        invalid = ~torch.isfinite(theta_new_all).all(dim=(1, 2))
        if invalid.any(): theta_new_all[invalid] = data['theta_all_prior'][invalid].unsqueeze(-1)
        W_new = theta_new_all[:, :data['fan_in'], 0]; b_new = theta_new_all[:, data['fan_in'], 0]
        theta_flat = theta_current.squeeze()
        theta_flat[data['W_start']:data['W_start']+data['nd_layer']['W_len']] = W_new.reshape(-1)
        theta_flat[data['b_start']:data['b_start']+data['nd_layer']['b_len']] = b_new
        theta_current = theta_flat.view(-1, 1)
        new_S_info.append(S_new_all.permute(1, 2, 0)); total_loss += data['loss']
    return theta_current, new_S_info, total_loss / layer_count

# =========================================================================
# 6. ★★★ [F] VectorEnv Training Loop ★★★
# =========================================================================
class RealTimePlot:
    def __init__(self, method_name):
        self.method_name = method_name
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 4))
        self.line1, = self.ax[0].plot([], [], 'b-', alpha=0.3)
        self.line1_ma, = self.ax[0].plot([], [], 'b-', linewidth=2)
        self.ax[0].axhline(y=195, color='g', linestyle='--')
        self.ax[0].set_ylim(0, 520)
        self.ax[0].set_title(f'Reward ({method_name})')
        self.line2, = self.ax[1].plot([], [], 'r-', linewidth=2)
        self.ax[1].set_title('Loss (Linear Scale)')
        self.line3, = self.ax[2].plot([], [], 'g-', linewidth=2)
        self.ax[2].set_ylim(0, cfg.p_init_max * 1.1)
        self.ax[2].set_title('Adaptive P_init')
        plt.tight_layout()
        self.rewards, self.losses, self.p_inits = [], [], []
    def add_data(self, r, l, p):
        self.rewards.append(r); self.losses.append(max(l, 1e-10)); self.p_inits.append(p)
    def refresh(self):
        self.line1.set_data(range(len(self.rewards)), self.rewards)
        if len(self.rewards)>=20:
            self.line1_ma.set_data(range(19, len(self.rewards)), np.convolve(self.rewards, np.ones(20)/20, 'valid'))
        self.line2.set_data(range(len(self.losses)), self.losses)
        self.line3.set_data(range(len(self.p_inits)), self.p_inits)
        for ax in self.ax: ax.relim(); ax.autoscale_view()
        plt.savefig(f'{self.method_name}_current.png', dpi=100)

def train(method: str):
    set_all_seeds(cfg.seed)
    
    # ★ [F] SyncVectorEnv: N개 CartPole 동시 실행
    num_envs = cfg.num_envs
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(cfg.env_name) for _ in range(num_envs)]
    )
    dimS = env.single_observation_space.shape[0]
    nA = env.single_action_space.n
    info = create_network_info(dimS, nA, cfg)
    
    method_names = {'full_vector': 'Full Vector', 'layer_decoupled': 'Layer Decoupled', 'node_decoupled': 'Node Decoupled'}
    explore_names = {'epsilon_greedy': 'EG', 'thompson_sampling': 'TS'}
    sf_map = {'full_vector': cfg.scale_factor_fv, 'layer_decoupled': cfg.scale_factor_ld, 'node_decoupled': cfg.scale_factor_nd}
    tau_map = {'full_vector': cfg.tau_fv, 'layer_decoupled': cfg.tau_ld, 'node_decoupled': cfg.tau_nd}
    r_map = {'full_vector': cfg.r_std_fv, 'layer_decoupled': cfg.r_std_ld, 'node_decoupled': cfg.r_std_nd}
    print(f"\nTraining {method_names[method]} + {explore_names[cfg.exploration]} | "
          f"Params: {info['total_params']} | Envs: {num_envs} | "
          f"R={r_map[method]} | τ={tau_map[method]} | SF={sf_map[method]}")
    
    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    nd_cache = NDCache(info, cfg, cfg.device) if method == 'node_decoupled' else None
    
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size, 
          'normalizer': normalizer, 'device': cfg.device}
    if method == 'full_vector':
        sp['Wm'], sp['Wc'], sp['gamma_sigma'] = compute_ut_weights(sp['n_x'], cfg.alpha, cfg.beta, cfg.kappa, cfg.device)
    
    theta = initialize_theta(info, cfg.device).view(-1, 1)
    theta_target = theta.clone()
    n_x = info['total_params']
    
    if method == 'full_vector': S_info = None
    elif method == 'layer_decoupled': layer_S_info = [None] * info['num_ld_layers']
    elif method == 'node_decoupled':
        neuron_S_info = [1e-6 * nd_cache.get(L)['eye_n_per'].unsqueeze(-1).expand(-1, -1, nd['fan_out']).clone() 
                         for L, nd in enumerate(info['nd_layers'])]
    
    if cfg.exploration == 'thompson_sampling':
        if method == 'full_vector':
            S_inv_t_cached = np.sqrt(cfg.p_init_max) * torch.eye(n_x, dtype=DTYPE, device=cfg.device)
        elif method == 'layer_decoupled':
            layer_S_inv_t_cached = [np.sqrt(cfg.p_init_max) * torch.eye(info['ld_layers'][L]['n_params'], dtype=DTYPE, device=cfg.device) 
                                    for L in range(info['num_ld_layers'])]
        elif method == 'node_decoupled':
            neuron_S_inv_t_cached = [np.sqrt(cfg.p_init_max) * torch.eye(nd_cache.get(L)['n_per'], dtype=DTYPE, device=cfg.device).unsqueeze(0).expand(nd_cache.get(L)['fan_out'], -1, -1).clone() 
                                     for L in range(info['num_nd_layers'])]
    
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    batch_hist = deque(maxlen=cfg.N_horizon)
    plotter = RealTimePlot(f"{method_names[method]}+{explore_names[cfg.exploration]}")
    
    scale_factor = sf_map[method]
    tau = tau_map[method]
    steps_done = 0
    completed_episodes = 0
    
    # ★ [F] 환경별 상태 추적
    running_rewards = np.zeros(num_envs)    # 각 환경의 현재 에피소드 누적 reward
    running_losses = [[] for _ in range(num_envs)]  # 각 환경의 현재 에피소드 loss
    
    # ★ [F] VectorEnv 초기 reset
    states, _ = env.reset(seed=cfg.seed)    # states: [num_envs, dimS]
    
    start_time = time.time()
    
    while completed_episodes < cfg.max_episodes:
        steps_done += 1
        global_steps = steps_done * num_envs
        # ── Adaptive P ──
        recent_rewards = plotter.rewards[-cfg.adaptive_window:] if plotter.rewards else []
        current_score = np.mean(recent_rewards) if recent_rewards else 0
        gap = max(0.0, min(1.0, 1.0 - current_score / cfg.max_steps))
        p_init = cfg.p_init_min + (cfg.p_init_max - cfg.p_init_min) * gap
        ts_temperature = np.sqrt(p_init)
        
        # ── ★ [F] Vectorized Action Selection ──
        with torch.no_grad():
            # states: [num_envs, dimS] → s_t: [dimS, num_envs]
            s_t = torch.tensor(states, dtype=DTYPE, device=cfg.device).t()
            if normalizer: s_t = normalizer.normalize(s_t)
            
            if cfg.exploration == 'epsilon_greedy':
                #eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-steps_done / cfg.eps_decay_steps)
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-global_steps / cfg.eps_decay_steps)
                # forward_single로 Q값 계산 (theta 1개, state N개 → Q: [nA, N])
                q_theta = theta.squeeze() if cfg.exploration == 'epsilon_greedy' else theta_explored
                q_values = forward_single(q_theta, info, s_t)  # [nA, num_envs]
                greedy_actions = q_values.squeeze().argmax(dim=0).cpu().numpy()  # [num_envs]
                
                # 환경별 독립적 epsilon-greedy
                rand_mask = np.random.rand(num_envs) < eps
                random_actions = np.random.randint(0, nA, size=num_envs)
                actions = np.where(rand_mask, random_actions, greedy_actions)
            
            elif cfg.exploration == 'thompson_sampling':
                if method == 'full_vector':
                    theta_explored = ts_sample_theta_fv(theta, S_inv_t_cached, ts_temperature, cfg.exploration_scale, n_x, cfg.device)
                elif method == 'layer_decoupled':
                    theta_explored = ts_sample_theta_ld(theta, layer_S_inv_t_cached, ts_temperature, cfg.exploration_scale, info, cfg.device)
                elif method == 'node_decoupled':
                    theta_explored = ts_sample_theta_nd(theta, neuron_S_inv_t_cached, ts_temperature, cfg.exploration_scale, info, nd_cache, cfg.device)
                
                q_values = forward_single(theta_explored, info, s_t)  # [nA, num_envs]
                actions = q_values.squeeze().argmax(dim=0).cpu().numpy()
        
        # ── ★ [F] Vectorized env.step ──
        next_states, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        
        # ── ★ [F] Batch push to buffer ──
        s_tensor = torch.tensor(states, dtype=DTYPE, device=cfg.device)
        a_tensor = torch.tensor(actions, dtype=torch.long, device=cfg.device)
        r_tensor = torch.tensor(rewards / scale_factor, dtype=DTYPE, device=cfg.device)
        ns_tensor = torch.tensor(next_states, dtype=DTYPE, device=cfg.device)
        d_tensor = torch.tensor(dones, dtype=DTYPE, device=cfg.device)
        buffer.push_batch(s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor)
        
        # ── 환경별 reward 누적 ──
        running_rewards += rewards
        
        # ── ★ [F] Done된 환경의 에피소드 기록 ──
        for i in range(num_envs):
            if dones[i]:
                ep_reward = running_rewards[i]
                avg_loss = np.mean(running_losses[i]) if running_losses[i] else 0
                plotter.add_data(ep_reward, avg_loss, p_init)
                completed_episodes += 1
                running_rewards[i] = 0
                running_losses[i] = []
                
                # 로그 출력 (10 에피소드마다)
                if completed_episodes % cfg.plot_interval == 0:
                    plotter.refresh()
                if completed_episodes % 10 == 0:
                    recent = np.mean(plotter.rewards[-20:]) if len(plotter.rewards) >= 20 else np.mean(plotter.rewards)
                    elapsed = time.time() - start_time
                    print(f"Ep {completed_episodes:3d} | Reward: {ep_reward:6.1f} | Avg20: {recent:6.1f} | "
                          f"Loss: {avg_loss:.4f} | P_init: {p_init:.4f} | Steps: {steps_done} | Time: {elapsed:.1f}s")
                
                if completed_episodes >= cfg.max_episodes:
                    break
        
        # VectorEnv는 done된 환경을 자동 reset하므로 next_states가 이미 새 에피소드의 state
        states = next_states
        
        # ── SRRHUIF 학습 (★ 기존과 완전 동일) ──
        if buffer.current_size >= cfg.batch_size*4:
            batch = buffer.sample_batch(cfg.batch_size)
            batch_hist.append(batch)
            
            if len(batch_hist) == cfg.N_horizon:
                for h in range(cfg.N_horizon):
                    is_first = (h == 0)
                    if method == 'full_vector':
                        theta, S_info, l_val = srrhuif_step_full_vector(theta, theta_target, S_info, batch_hist[h], sp, is_first, p_init)
                    elif method == 'layer_decoupled':
                        theta, layer_S_info, l_val = srrhuif_step_layer_decoupled(theta, theta_target, layer_S_info, batch_hist[h], sp, is_first, p_init)
                    elif method == 'node_decoupled':
                        theta, neuron_S_info, l_val = srrhuif_step_nd(theta, theta_target, neuron_S_info, batch_hist[h], sp, is_first, p_init, nd_cache)
                    
                    # 모든 활성 환경에 loss 기록
                    for i in range(num_envs):
                        running_losses[i].append(l_val)
                
                theta_target = (1.0 - tau) * theta_target + tau * theta
                
                if cfg.exploration == 'thompson_sampling':
                    if method == 'full_vector' and S_info is not None:
                        S_inv_t_cached = compute_S_inv_t(S_info, n_x, cfg.device)
                    elif method == 'layer_decoupled':
                        for L in range(info['num_ld_layers']):
                            if layer_S_info[L] is not None:
                                layer_S_inv_t_cached[L] = compute_S_inv_t(layer_S_info[L], info['ld_layers'][L]['n_params'], cfg.device)
                    elif method == 'node_decoupled':
                        for L in range(info['num_nd_layers']):
                            if neuron_S_info[L] is not None:
                                lc = nd_cache.get(L)
                                neuron_S_inv_t_cached[L] = compute_S_inv_t_batch(
                                    neuron_S_info[L].permute(2, 0, 1), lc['n_per'], cfg.device)
    
    env.close()
    plotter.refresh()
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete | Episodes: {completed_episodes} | Steps: {steps_done} | Time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # =========================================================================
    # 7. ★ [수정] 시연용 평가 및 GIF/MP4 영상 저장 ★
    # =========================================================================
    print("\nStarting Evaluation and Recording with trained parameters...")
    try:
        # render_mode를 "rgb_array"로 변경하여 화면 데이터를 배열로 추출
        eval_env = gym.make(cfg.env_name, render_mode="rgb_array")
        state, _ = eval_env.reset(seed=cfg.seed)
        
        eval_reward = 0
        done = False
        step_count = 0
        frames = [] # 화면 픽셀 데이터를 모아둘 리스트
        
        while not done and step_count < cfg.max_steps:
            # 1. 현재 프레임 캡처 후 리스트에 저장
            frames.append(eval_env.render())
            
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=DTYPE, device=cfg.device).unsqueeze(1)
                if normalizer: 
                    s_t = normalizer.normalize(s_t)
                
                # 2. 오직 학습된 theta만 사용하여 행동 선택
                q_values = forward_single(theta, info, s_t)
                action = q_values.squeeze().argmax().item()
                
            state, reward, terminated, truncated, _ = eval_env.step(action)
            eval_reward += reward
            step_count += 1
            
            done = terminated or truncated
            
        print(f"Evaluation finished! Total Reward: {eval_reward} | Steps: {step_count}")
        eval_env.close()
        
        # 3. 캡처한 프레임들을 모아서 파일로 저장
        save_path = f"CartPole_ND_2000params_Score{int(eval_reward)}.gif"
        print(f"Saving video to {save_path}... Please wait.")
        
        # fps=30으로 설정하여 부드러운 GIF 생성 (mp4로 저장하려면 확장자만 .mp4로 변경)
        imageio.mimsave(save_path, frames, fps=30)
        print("Video saved successfully!")
        
    except Exception as e:
        print(f"Recording failed: {e}")

    return plotter.rewards, plotter.p_inits

if __name__ == "__main__":
    train('node_decoupled')
    #train('full_vector')
