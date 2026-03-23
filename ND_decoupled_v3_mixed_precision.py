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

matplotlib.use('Agg')

"""
=========================================================================
SRRHUIF-D3QN v3: Mixed Precision (A+B+C+D+E)
=========================================================================
  기존 (A~D 유지):
    [A] Compile-Safe Math   [B] NDCache Pre-compute
    [C] torch.compile       [D] Unified Forward (1회 forward_bmm)

  ★ [E] 3-Tier Mixed Precision:
    ┌─────────────────────────────────────────────────────────────┐
    │ Tier 1 — FP32 (Forward Zone)                                │
    │   forward_bmm/single 내부 연산, Z_sigma, z_hat, Z_dev,      │
    │   P_xz 모두 FP32. TF32 텐서코어 활용 가능.                  │
    │                                                             │
    │ ★ Cast Boundary: P_xz(FP32) → FP64, z_hat(FP32) → FP64     │
    │                                                             │
    │ Tier 2 — FP64 (Filter Zone)                                 │
    │   HT = Y_pred(64) @ P_xz(64), residual = z_meas(64)-z_hat(64)│
    │   QR, solve_triangular, 정보행렬 업데이트 전부 FP64          │
    │                                                             │
    │ Tier 3 — FP64 (Storage)                                     │
    │   theta, S_info, Y_pred, y_pred 등 필터 상태 변수           │
    └─────────────────────────────────────────────────────────────┘

  캐스팅 포인트: forward 입력(64→32) + HT 경계(P_xz 32→64, z_hat 32→64)
  Unified Buffer: FP32로 사전 할당 → 매 step cast 제거
=========================================================================
"""

print("="*70)
print(f"SRRHUIF-D3QN v3 (Mixed Precision) | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    # ★ [E] TF32 활성화: FP32 matmul에 텐서코어 활용
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
DTYPE_FWD = torch.float32  # ★ [E] Forward zone precision
JITTER = 1e-6

# =========================================================================
# 1. Configuration
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_episodes: int = 200
    max_steps: int = 500
    batch_size: int = 128      
    buffer_size: int = 100000
    
    shared_layers: List[int] = field(default_factory=lambda: [16,16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])
    
    gamma: float = 0.94     
    
    scale_factor_fv: float = 1.0
    scale_factor_ld: float = 1.0
    scale_factor_nd: float = 1.0
    
    tau_fv: float = 0.005
    tau_ld: float = 0.005
    tau_nd: float = 0.005
    
    exploration: str = "epsilon_greedy"
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000
    exploration_scale: float = 3.0
    
    N_horizon: int = 5  
    q_std: float = 5e-3
    r_std_fv: float = 2.9
    r_std_ld: float = 3 
    r_std_nd: float = 2
    ld_global_denom: bool = False   
    nd_global_denom: bool = False   
    
    update_interval: int = 1   # 몇 step마다 SRRHUIF 업데이트할지

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
    
    def __post_init__(self):
        self.r_inv_sqrt_fv = 1.0 / self.r_std_fv
        self.r_inv_fv      = 1.0 / (self.r_std_fv ** 2)
        self.r_inv_sqrt_ld = 1.0 / self.r_std_ld
        self.r_inv_ld      = 1.0 / (self.r_std_ld ** 2)
        self.r_inv_sqrt_nd = 1.0 / self.r_std_nd
        self.r_inv_nd      = 1.0 / (self.r_std_nd ** 2)

cfg = Config()

# =========================================================================
# 2. Network Info & Cache
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
    """[B]+[D]+★[E]: Pre-compute + Unified Buffer (FP32) + FP32 캐시"""
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
            
            # FP64 캐시 (필터 수학용)
            eye_n_per = torch.eye(n_per, dtype=DTYPE, device=device)
            eye_n_per_batch = eye_n_per.unsqueeze(0).expand(fan_out, -1, -1).clone()
            
            lamb = cfg.alpha**2 * (n_per + cfg.kappa) - n_per
            gamma = float(np.sqrt(n_per + lamb))
            Wm = torch.zeros(2*n_per+1, dtype=DTYPE, device=device)
            Wc = torch.zeros(2*n_per+1, dtype=DTYPE, device=device)
            Wm[0] = lamb / (n_per + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha**2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (n_per + lamb)
            
            # [B] FP64 사전 할당 (Time/Meas Update용)
            S_Q_cached = cfg.q_std * eye_n_per_batch.clone()
            
            # ★ [E] FP32 사전 할당 (HT 계산 FP32 zone용)
            Wm_col_f32 = Wm.to(DTYPE_FWD).view(1, -1, 1).expand(fan_out, -1, -1).clone()
            Wc_f32 = Wc.to(DTYPE_FWD)
            zero_col_f32 = torch.zeros(fan_out, n_per, 1, dtype=DTYPE_FWD, device=device)
            
            self.layers[L] = {
                'w_col_idx': w_col_idx, 'b_col_idx': b_col_idx,
                'eye_n_per': eye_n_per, 'eye_n_per_batch': eye_n_per_batch,
                'Wm': Wm, 'Wc': Wc, 'gamma': gamma,
                'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per, 'num_sigma': num_sigma,
                'S_Q_cached': S_Q_cached,
                # ★ [E] FP32 캐시
                'Wm_col_f32': Wm_col_f32,
                'Wc_f32': Wc_f32,
                'zero_col_f32': zero_col_f32,
            }
        
        # ★ [D]+[E] Unified Forward Buffer — FP32로 할당
        #   FP64 theta → FP32 copy_ (자동 캐스팅, 매 step 새 텐서 할당 없음)
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], 
                                          dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards
        
        bytes_per_elem = 4  # FP32
        print(f"  [D+E] Unified FP32 Buffer: {total_forwards} × {info['total_params']} "
              f"= {total_forwards * info['total_params'] * bytes_per_elem / 1024:.1f} KB")
    
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
# 3. Forward Functions — ★ [E] FP32 내부 연산
# =========================================================================
def forward_single(theta, info, x):
    """Action Selection용 — FP32 내부, FP64 반환"""
    theta = theta.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    x = x.to(DTYPE_FWD)
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
    return (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)

def forward_bmm(thetas, info, x):
    """★ [E] FP32 내부 연산, FP32 반환
    입력이 FP64이면 자동 캐스팅, FP32이면 no-op"""
    thetas = thetas.to(DTYPE_FWD)
    x = x.to(DTYPE_FWD)
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
    return v + (a - a.mean(dim=1, keepdim=True))  # ★ FP32 반환 (cast 없음)

def forward_3part_ld(theta_current, info, x, opt_layer_idx, theta_sigma_L):
    """LD forward — forward_bmm이 FP32로 반환"""
    num_sigma = theta_sigma_L.shape[0]
    thetas = theta_current.squeeze().unsqueeze(0).expand(num_sigma, -1).clone()
    ld_layer = info['ld_layers'][opt_layer_idx]
    start, end = ld_layer['param_start'], ld_layer['param_end']
    thetas[:, start:end] = theta_sigma_L
    return forward_bmm(thetas, info, x)

# =========================================================================
# Pre-allocated GPU Tensor Replay Buffer
# =========================================================================
class TensorReplayBuffer:
    def __init__(self, capacity: int, dimS: int, device: str):
        self.capacity, self.count, self.device = capacity, 0, device
        self.S      = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A      = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R      = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term   = torch.zeros(capacity, dtype=DTYPE, device=device)
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

# =========================================================================
# 4. Math Utilities — [A] Compile-Safe (FP64 고정)
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

# =========================================================================
# [C] ND Compiled Core Functions
# =========================================================================
def _nd_time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    """Time Update (FP64) — 필터 수학"""
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
    """★ [E] Mixed Precision HT 계산 (Compilable)
    
    FP32 zone: z_hat, Z_dev, X_dev, P_xz (forward 출력 정밀도 유지)
    Cast boundary: P_xz → FP64, z_hat → FP64
    FP64 zone: residual, HT (필터 정밀도 보장)
    
    Args:
        Z_sigma_T_f32:      [fan_out, batch_sz, num_sigma] FP32 (forward 출력)
        Wm_col_f32:         [fan_out, num_sigma, 1] FP32 (캐시)
        Wc_f32:             [num_sigma] FP32 (캐시)
        zero_col_f32:       [fan_out, n_per, 1] FP32 (캐시)
        scaled_P_f32:       [fan_out, n_per, n_per] FP32 (caller가 캐스팅)
        z_measured_exp_f64: [fan_out, batch_sz, 1] FP64
        Y_pred_f64:         [fan_out, n_per, n_per] FP64
    """
    # ── FP32 zone ──
    z_hat_f32 = torch.bmm(Z_sigma_T_f32, Wm_col_f32)          # [fan_out, batch_sz, 1]
    Z_dev_f32 = Z_sigma_T_f32 - z_hat_f32                      # [fan_out, batch_sz, num_sigma]
    X_dev_f32 = torch.cat([zero_col_f32, scaled_P_f32, -scaled_P_f32], dim=2)
    P_xz_f32 = torch.bmm(
        X_dev_f32 * Wc_f32.view(1, 1, -1),                     # [fan_out, n_per, num_sigma]
        Z_dev_f32.transpose(1, 2)                               # [fan_out, num_sigma, batch_sz]
    )                                                            # [fan_out, n_per, batch_sz]
    
    # ── Cast boundary → FP64 ──
    z_hat_f64 = z_hat_f32.to(torch.float64)
    residual_all = z_measured_exp_f64 - z_hat_f64               # FP64
    HT_all = torch.bmm(Y_pred_f64, P_xz_f32.to(torch.float64))  # FP64
    
    return HT_all, residual_all, z_hat_f64

def _nd_meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all,
                          r_inv_sqrt, r_inv, eye_batch):
    """Measurement Update (FP64) — 필터 수학"""
    combined = torch.cat([S_pred, HT_all * r_inv_sqrt], dim=2)
    S_new_all = tria_operation_batch(combined)
    innov = residual_all + torch.bmm(HT_all.transpose(1, 2), theta_3d)
    y_new_all = y_pred + torch.bmm(HT_all, r_inv * innov)
    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    return theta_new_all, S_new_all

# =========================================================================
# [C] torch.compile
# =========================================================================
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

# =========================================================================
# Thompson Sampling
# =========================================================================
def compute_S_inv_t(S, n, device):
    eye = torch.eye(n, dtype=S.dtype, device=device)
    S_inv = torch.linalg.solve_triangular(S + JITTER * eye, eye, upper=False)
    S_inv_t = S_inv.t()
    return torch.where(torch.isfinite(S_inv_t), S_inv_t, eye)

def compute_S_inv_t_batch(S_batch, n, device):
    eye = torch.eye(n, dtype=S_batch.dtype, device=device).unsqueeze(0).expand(S_batch.shape[0], -1, -1)
    S_inv = torch.linalg.solve_triangular(S_batch + JITTER * eye, eye, upper=False)
    S_inv_t = S_inv.transpose(-2, -1)
    return torch.where(torch.isfinite(S_inv_t), S_inv_t, eye)

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
        nd_layer = info['nd_layers'][L]
        lc = nd_cache.get(L)
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
    Wm[0] = lamb / (n + lamb)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 0.5 / (n + lamb)
    return Wm, Wc, float(np.sqrt(n + lamb))

def generate_sigma_points(theta_prior, P_sqrt, gamma, n, device):
    theta_flat = theta_prior.squeeze()
    scaled_P = gamma * P_sqrt
    X_sigma = torch.cat([theta_flat.unsqueeze(0),
        theta_flat.unsqueeze(0) + scaled_P.t(),
        theta_flat.unsqueeze(0) - scaled_P.t()], dim=0)
    return X_sigma, scaled_P

# =========================================================================
# 5. Step Functions
# =========================================================================

# ── FV/LD 공통: forward_bmm이 FP32를 반환하므로 Z_sigma를 FP64로 캐스팅 ──

@torch.no_grad()
def srrhuif_step_full_vector(theta_current, theta_target, S_info, batch, sp, is_first, p_init_val):
    device, n_x, batch_sz, info = sp['device'], sp['n_x'], sp['batch_sz'], sp['info']
    theta_pred = theta_target if is_first else theta_current

    P_sqrt_prev = p_init_val * torch.eye(n_x, dtype=DTYPE, device=device) if is_first else safe_inv_tril(S_info, n_x, device)
    S_Q = cfg.q_std * torch.eye(n_x, dtype=DTYPE, device=device)
    P_sqrt_pred = tria_operation(torch.cat([P_sqrt_prev, S_Q], dim=1))
    S_pred = safe_inv_tril(P_sqrt_pred, n_x, device)
    Y_pred = S_pred @ S_pred.t()
    y_pred = Y_pred @ theta_pred
    X_sigma, scaled_P = generate_sigma_points(theta_pred, P_sqrt_pred, sp['gamma_sigma'], n_x, device)
    
    s_batch = batch['s'].t()
    if sp.get('normalizer'): s_batch = sp['normalizer'].normalize(s_batch)
    Q_all_f32 = forward_bmm(X_sigma, info, s_batch)                          # ★ FP32
    Z_sigma = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)].t().to(DTYPE)  # ★ → FP64
    z_hat = Z_sigma @ sp['Wm'].view(-1, 1)
    
    s_next = batch['s_next'].t()
    if sp.get('normalizer'): s_next = sp['normalizer'].normalize(s_next)
    """
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both_f32 = forward_bmm(thetas_pair, info, s_next)                      # ★ FP32
    a_best_next = Q_both_f32[0].argmax(dim=0)
    q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)  # ★ → FP64
    """
    
    # ── DDQN Target ──
    if is_first:
        Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next)
        q_val_next = Q_tgt_f32[0].max(dim=0).values.to(DTYPE)
    else:
        thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
        Q_both_f32 = forward_bmm(thetas_pair, info, s_next)
        a_best_next = Q_both_f32[0].argmax(dim=0)
        q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)
    
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    
    residual = z_measured - z_hat
    X_dev = torch.cat([torch.zeros(n_x, 1, dtype=DTYPE, device=device), scaled_P, -scaled_P], dim=1)
    P_xz = (X_dev * sp['Wc']) @ (Z_sigma - z_hat).t()
    HT = Y_pred @ P_xz
    S_new = tria_operation(torch.cat([S_pred, HT * cfg.r_inv_sqrt_fv], dim=1))
    y_new = y_pred + HT @ (cfg.r_inv_fv * (residual + HT.t() @ theta_pred))
    theta_new = robust_solve_spd(S_new, y_new, n_x, device)
    if not torch.isfinite(theta_new).all(): theta_new = theta_pred.clone()
    return theta_new, S_new, torch.mean(residual**2).item()

@torch.no_grad()
def srrhuif_step_layer_decoupled(theta_current_in, theta_target, layer_S_info, batch, sp, is_first, p_init_val):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info, total_loss = [], 0.0
    
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both_f32 = forward_bmm(thetas_pair, info, s_next)                      # ★ FP32
    a_best_next = Q_both_f32[0].argmax(dim=0)
    q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)  # → FP64
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    
    layer_data, P_zz_sum = [], 0.0
    for L in range(info['num_ld_layers']):
        ld_layer = info['ld_layers'][L]
        n_L, start, end = ld_layer['n_params'], ld_layer['param_start'], ld_layer['param_end']
        theta_L_prior = theta_prior.squeeze()[start:end].view(-1, 1)
        S_L = layer_S_info[L]
        P_sqrt_prev = p_init_val * torch.eye(n_L, dtype=DTYPE, device=device) if is_first or S_L is None else safe_inv_tril(S_L, n_L, device)
        S_Q = cfg.q_std * torch.eye(n_L, dtype=DTYPE, device=device)
        P_sqrt_pred = tria_operation(torch.cat([P_sqrt_prev, S_Q], dim=1))
        S_pred = safe_inv_tril(P_sqrt_pred, n_L, device)
        Y_pred = S_pred @ S_pred.t(); y_pred = Y_pred @ theta_L_prior
        Wm_L, Wc_L, gamma_L = compute_ut_weights(n_L, cfg.alpha, cfg.beta, cfg.kappa, device)
        X_sigma_L, scaled_P_L = generate_sigma_points(theta_L_prior, P_sqrt_pred, gamma_L, n_L, device)
        Q_all_f32 = forward_3part_ld(theta_current, info, s_batch, L, X_sigma_L)   # ★ FP32
        Z_sigma = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)].t().to(DTYPE)  # → FP64
        z_hat = Z_sigma @ Wm_L.view(-1, 1)
        residual = z_measured - z_hat
        X_dev = torch.cat([torch.zeros(n_L, 1, dtype=DTYPE, device=device), scaled_P_L, -scaled_P_L], dim=1)
        HT = Y_pred @ ((X_dev * Wc_L) @ (Z_sigma - z_hat).t())
        Z_dev_out = Z_sigma - z_hat
        P_zz_L = (Wc_L.view(1, -1) * Z_dev_out ** 2).sum(dim=1).mean().item()
        P_zz_sum += P_zz_L
        layer_data.append({'n_L': n_L, 'start': start, 'end': end, 'theta_L_prior': theta_L_prior,
            'S_pred': S_pred, 'y_pred': y_pred, 'HT': HT, 'residual': residual,
            'loss': torch.mean(residual**2).item(), 'P_zz_L': P_zz_L})
    
    for L, data in enumerate(layer_data):
        if cfg.ld_global_denom:
            R_eff_L = max(cfg.r_std_ld ** 2 + (P_zz_sum - data['P_zz_L']), 1e-10)
            r_inv_sqrt_L, r_inv_L = 1.0 / np.sqrt(R_eff_L), 1.0 / R_eff_L
        else:
            r_inv_sqrt_L, r_inv_L = cfg.r_inv_sqrt_ld, cfg.r_inv_ld
        S_new = tria_operation(torch.cat([data['S_pred'], data['HT'] * r_inv_sqrt_L], dim=1))
        y_new = data['y_pred'] + data['HT'] @ (r_inv_L * (data['residual'] + data['HT'].t() @ data['theta_L_prior']))
        theta_L_new = robust_solve_spd(S_new, y_new, data['n_L'], device)
        if not torch.isfinite(theta_L_new).all(): theta_L_new = data['theta_L_prior'].clone()
        theta_current.squeeze()[data['start']:data['end']] = theta_L_new.squeeze()
        new_S_info.append(S_new); total_loss += data['loss']
    return theta_current, new_S_info, total_loss / info['num_ld_layers']

# =========================================================================
# ★★★ [D+E] ND Step — Unified Forward (FP32) + Mixed Precision ★★★
# =========================================================================
@torch.no_grad()
def srrhuif_step_nd(theta_current_in, theta_target, neuron_S_info, batch, sp, is_first, p_init_val, nd_cache):
    """
    Precision 흐름:
      DDQN: forward_bmm(FP32) → q_val_next.to(FP64) → z_measured(FP64)
      Phase 1a: Time Update(FP64) → X_sigma(FP64) → .to(FP32) → unified(FP32)
      Phase 1b: forward_bmm(FP32) → Q_all(FP32)
      Phase 1c: _nd_compute_ht_core(FP32 zone → FP64 boundary) → HT(FP64), residual(FP64)
      Phase 2:  _nd_meas_update_core(FP64) → theta_new(FP64)
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info, total_loss, layer_count = [], 0.0, 0
    
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    # ── DDQN Target ──
    """
    thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0)
    Q_both_f32 = forward_bmm(thetas_pair, info, s_next)                      # ★ FP32
    a_best_next = Q_both_f32[0].argmax(dim=0)
    q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)  # → FP64
    """
    if is_first:
        Q_tgt_f32 = forward_bmm(theta_target.squeeze().unsqueeze(0), info, s_next) # 첫스탭에서 target 값은 Target_Q_network로만 계산 (Standard DQN)
        q_val_next = Q_tgt_f32[0].max(dim=0).values.to(DTYPE)
    else:
        thetas_pair = torch.stack([theta_current.squeeze(), theta_target.squeeze()], dim=0) 
        Q_both_f32 = forward_bmm(thetas_pair, info, s_next)
        a_best_next = Q_both_f32[0].argmax(dim=0) # 이후 action은 이전 스탭에서 추정한 Theta_current로 계산 (DDQN)
        q_val_next = Q_both_f32[1][a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)

    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    
    # =====================================================================
    # Phase 1a: Time Update (FP64) + Scatter to FP32 unified buffer
    # =====================================================================
    unified = nd_cache.unified_thetas                                          # ★ FP32 buffer
    unified[:] = theta_current.squeeze().to(DTYPE_FWD)                         # FP64 → FP32 copy
    
    time_data = []
    for L in range(info['num_nd_layers']):
        nd_layer = info['nd_layers'][L]
        lc = nd_cache.get(L)
        fan_in, fan_out, n_per = lc['fan_in'], lc['fan_out'], lc['n_per']
        W_start, b_start = nd_layer['W_start'], nd_layer['b_start']
        
        W_prior = theta_prior.squeeze()[W_start:W_start+nd_layer['W_len']].view(fan_out, fan_in)
        b_prior = theta_prior.squeeze()[b_start:b_start+nd_layer['b_len']]
        theta_all_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1)
        theta_all_prior_3d = theta_all_prior.unsqueeze(-1)
        
        S_3d = neuron_S_info[L]
        if is_first or S_3d is None:
            P_sqrt_prev = np.sqrt(p_init_val) * lc['eye_n_per_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_n_per_batch'])
        
        # [C] COMPILED: Time Update (FP64)
        S_pred, Y_pred, y_pred, X_sigma_all, scaled_P = _nd_time_update_core(
            theta_all_prior_3d, P_sqrt_prev, lc['S_Q_cached'], lc['eye_n_per_batch'], lc['gamma'])
        
        # ★ [E] FP64 sigma → FP32 scatter
        X_sigma_f32 = X_sigma_all.to(DTYPE_FWD)
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :fan_in])
        layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, fan_in:fan_in+1])
        
        time_data.append({
            'nd_layer': nd_layer, 'lc': lc,
            'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per,
            'W_start': W_start, 'b_start': b_start,
            'theta_all_prior': theta_all_prior, 'theta_all_prior_3d': theta_all_prior_3d,
            'S_pred': S_pred, 'Y_pred': Y_pred, 'y_pred': y_pred, 'scaled_P': scaled_P,
        })
    
    # =====================================================================
    # Phase 1b: ★ Single forward_bmm (FP32)
    # =====================================================================
    Q_all_f32 = forward_bmm(unified, info, s_batch)  # ★ FP32 in, FP32 out
    
    # =====================================================================
    # Phase 1c: HT Computation — ★ [E] Mixed Precision Core
    # =====================================================================
    layer_data = []
    for L in range(info['num_nd_layers']):
        td = time_data[L]
        lc = td['lc']
        fan_out = td['fan_out']
        
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)  # FP32
        
        z_measured_exp = z_measured.unsqueeze(0).expand(fan_out, -1, -1)  # FP64
        
        # ★ [E] COMPILED: FP32 zone → FP64 boundary
        HT_all, residual_all, z_hat_f64 = _nd_compute_ht_core(
            Z_sigma_T_f32,                          # FP32
            lc['Wm_col_f32'],                       # FP32 (캐시)
            lc['Wc_f32'],                            # FP32 (캐시)
            lc['zero_col_f32'],                      # FP32 (캐시)
            td['scaled_P'].to(DTYPE_FWD),            # FP64 → FP32 (1회/layer)
            z_measured_exp,                          # FP64
            td['Y_pred'])                            # FP64
        
        loss_L = torch.mean(residual_all**2)
        
        layer_data.append({
            'nd_layer': td['nd_layer'], 'lc': lc,
            'fan_in': td['fan_in'], 'fan_out': fan_out, 'n_per': td['n_per'],
            'W_start': td['W_start'], 'b_start': td['b_start'],
            'theta_all_prior': td['theta_all_prior'], 'theta_all_prior_3d': td['theta_all_prior_3d'],
            'S_pred': td['S_pred'], 'y_pred': td['y_pred'],
            'HT_all': HT_all, 'residual_all': residual_all, 'loss': loss_L,
        })
        layer_count += 1
    
    # =====================================================================
    # Phase 2: Measurement Update (FP64)
    # =====================================================================
    for L, data in enumerate(layer_data):
        theta_new_all, S_new_all = _nd_meas_update_core(
            data['S_pred'], data['y_pred'], data['HT_all'],
            data['theta_all_prior_3d'], data['residual_all'],
            cfg.r_inv_sqrt_nd, cfg.r_inv_nd, data['lc']['eye_n_per_batch'])
        
        invalid = ~torch.isfinite(theta_new_all).all(dim=(1, 2))
        if invalid.any():
            theta_new_all[invalid] = data['theta_all_prior'][invalid].unsqueeze(-1)
        
        W_new = theta_new_all[:, :data['fan_in'], 0]
        b_new = theta_new_all[:, data['fan_in'], 0]
        theta_flat = theta_current.squeeze()
        theta_flat[data['W_start']:data['W_start']+data['nd_layer']['W_len']] = W_new.reshape(-1)
        theta_flat[data['b_start']:data['b_start']+data['nd_layer']['b_len']] = b_new
        theta_current = theta_flat.view(-1, 1)
        
        new_S_info.append(S_new_all.permute(1, 2, 0))
        total_loss = torch.tensor(0.0, dtype=DTYPE, device=device)  # GPU 텐서
        total_loss = total_loss + data['loss']
    return theta_current, new_S_info, (total_loss / layer_count).item()

# =========================================================================
# 6. Training Loop
# =========================================================================
class RealTimePlot:
    def __init__(self, max_episodes, method_name, filename):
        self.method_name = method_name
        self.filename = filename
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 4))
        self.line1, = self.ax[0].plot([], [], 'b-', alpha=0.3)
        self.line1_ma, = self.ax[0].plot([], [], 'b-', linewidth=2)
        self.ax[0].axhline(y=195, color='g', linestyle='--')
        self.ax[0].set_xlim(0, max_episodes); self.ax[0].set_ylim(0, 520)
        self.ax[0].set_title(f'Reward ({method_name})')
        self.line2, = self.ax[1].plot([], [], 'r-', linewidth=2); self.ax[1].set_title('Loss')
        self.line3, = self.ax[2].plot([], [], 'g-', linewidth=2)
        self.ax[2].set_ylim(0, cfg.p_init_max * 1.1); self.ax[2].set_title('Adaptive P_init')
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
        plt.savefig(f'{self.filename}.png', dpi=100)

def train(method: str):
    set_all_seeds(cfg.seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    
    method_names = {'full_vector': 'Full Vector', 'layer_decoupled': 'Layer Decoupled', 'node_decoupled': 'Node Decoupled'}
    explore_names = {'epsilon_greedy': 'EG', 'thompson_sampling': 'TS'}
    sf_map = {'full_vector': cfg.scale_factor_fv, 'layer_decoupled': cfg.scale_factor_ld, 'node_decoupled': cfg.scale_factor_nd}
    tau_map = {'full_vector': cfg.tau_fv, 'layer_decoupled': cfg.tau_ld, 'node_decoupled': cfg.tau_nd}
    r_map = {'full_vector': cfg.r_std_fv, 'layer_decoupled': cfg.r_std_ld, 'node_decoupled': cfg.r_std_nd}
    print(f"\nTraining {method_names[method]} + {explore_names[cfg.exploration]} | Params: {info['total_params']} | R={r_map[method]} | τ={tau_map[method]} | SF={sf_map[method]}")
    
    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    nd_cache = NDCache(info, cfg, cfg.device) if method == 'node_decoupled' else None
    
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size, 'normalizer': normalizer, 'device': cfg.device}
    if method == 'full_vector':
        sp['Wm'], sp['Wc'], sp['gamma_sigma'] = compute_ut_weights(sp['n_x'], cfg.alpha, cfg.beta, cfg.kappa, cfg.device)
    
    theta = initialize_theta(info, cfg.device).view(-1, 1)
    theta_target = theta.clone()
    n_x = info['total_params']
    
    if method == 'full_vector': S_info = None
    elif method == 'layer_decoupled': layer_S_info = [None] * info['num_ld_layers']
    elif method == 'node_decoupled':
        neuron_S_info = [1e-6 * nd_cache.get(L)['eye_n_per'].unsqueeze(-1).expand(-1, -1, nd['fan_out']).clone() for L, nd in enumerate(info['nd_layers'])]
    
    if cfg.exploration == 'thompson_sampling':
        if method == 'full_vector':
            S_inv_t_cached = np.sqrt(cfg.p_init_max) * torch.eye(n_x, dtype=DTYPE, device=cfg.device)
        elif method == 'layer_decoupled':
            layer_S_inv_t_cached = [np.sqrt(cfg.p_init_max) * torch.eye(info['ld_layers'][L]['n_params'], dtype=DTYPE, device=cfg.device) for L in range(info['num_ld_layers'])]
        elif method == 'node_decoupled':
            neuron_S_inv_t_cached = [np.sqrt(cfg.p_init_max) * torch.eye(nd_cache.get(L)['n_per'], dtype=DTYPE, device=cfg.device).unsqueeze(0).expand(nd_cache.get(L)['fan_out'], -1, -1).clone() for L in range(info['num_nd_layers'])]
    
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)  # action selection용

    batch_hist = deque(maxlen=cfg.N_horizon)
    plot_title = f"{method_names[method]}+{explore_names[cfg.exploration]}"
    plot_filename = (f"{plot_title}_Q{cfg.q_std}_R{r_map[method]}_gamma{cfg.gamma}"
                 f"_batch{cfg.batch_size}_horizon{cfg.N_horizon}"
                 f"_Pmax{cfg.p_init_max}Pmin{cfg.p_init_min}"
                 f"_tau{tau_map[method]}_params{info['total_params']}")
    plotter = RealTimePlot(cfg.max_episodes, plot_title, plot_filename)
    steps_done = 0
    scale_factor, tau = sf_map[method], tau_map[method]
    
    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_r, ep_l, ep_start = 0, [], time.time()
        start_idx = max(0, len(plotter.rewards) - cfg.adaptive_window)
        current_score = np.mean(plotter.rewards[start_idx:]) if plotter.rewards else 0
        gap = max(0.0, min(1.0, 1.0 - current_score / cfg.max_steps))
        p_init = cfg.p_init_min + (cfg.p_init_max - cfg.p_init_min) * gap
        ts_temperature = np.sqrt(p_init)
        
        for t in range(cfg.max_steps):
            steps_done += 1
            if cfg.exploration == 'epsilon_greedy':
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-steps_done / cfg.eps_decay_steps)
                if np.random.rand() < eps: a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))  # CPU 텐서 생성 후 GPU buffer에 copy
                        s_t = s_t_buffer
                        if normalizer: s_t = normalizer.normalize(s_t)
                        a = forward_single(theta.squeeze(), info, s_t).squeeze().argmax().item()
            elif cfg.exploration == 'thompson_sampling':
                with torch.no_grad():
                    if method == 'full_vector': theta_explored = ts_sample_theta_fv(theta, S_inv_t_cached, ts_temperature, cfg.exploration_scale, n_x, cfg.device)
                    elif method == 'layer_decoupled': theta_explored = ts_sample_theta_ld(theta, layer_S_inv_t_cached, ts_temperature, cfg.exploration_scale, info, cfg.device)
                    elif method == 'node_decoupled': theta_explored = ts_sample_theta_nd(theta, neuron_S_inv_t_cached, ts_temperature, cfg.exploration_scale, info, nd_cache, cfg.device)
                    s_t = torch.tensor(s, dtype=DTYPE, device=cfg.device)
                    if normalizer: s_t = normalizer.normalize(s_t)
                    a = forward_single(theta_explored, info, s_t).squeeze().argmax().item()
            
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / scale_factor, ns, done or trunc)
            s, ep_r = ns, ep_r + r
            
            if buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                batch = buffer.sample_batch(cfg.batch_size)
                batch_hist.append(batch)
                if len(batch_hist) == cfg.N_horizon:
                    for h in range(cfg.N_horizon):
                        is_first = (h == 0)
                        if method == 'full_vector': theta, S_info, l_val = srrhuif_step_full_vector(theta, theta_target, S_info, batch_hist[h], sp, is_first, p_init)
                        elif method == 'layer_decoupled': theta, layer_S_info, l_val = srrhuif_step_layer_decoupled(theta, theta_target, layer_S_info, batch_hist[h], sp, is_first, p_init)
                        elif method == 'node_decoupled': theta, neuron_S_info, l_val = srrhuif_step_nd(theta, theta_target, neuron_S_info, batch_hist[h], sp, is_first, p_init, nd_cache)
                        ep_l.append(l_val)
                    theta_target = (1.0 - tau) * theta_target + tau * theta
                    if cfg.exploration == 'thompson_sampling':
                        if method == 'full_vector' and S_info is not None: S_inv_t_cached = compute_S_inv_t(S_info, n_x, cfg.device)
                        elif method == 'layer_decoupled':
                            for L in range(info['num_ld_layers']):
                                if layer_S_info[L] is not None: layer_S_inv_t_cached[L] = compute_S_inv_t(layer_S_info[L], info['ld_layers'][L]['n_params'], cfg.device)
                        elif method == 'node_decoupled':
                            for L in range(info['num_nd_layers']):
                                if neuron_S_info[L] is not None:
                                    neuron_S_inv_t_cached[L] = compute_S_inv_t_batch(neuron_S_info[L].permute(2, 0, 1), nd_cache.get(L)['n_per'], cfg.device)
            if done or trunc: break
        
        ep_time = time.time() - ep_start
        avg_l = np.mean(ep_l) if ep_l else 0
        plotter.add_data(ep_r, avg_l, p_init)
        if ep % cfg.plot_interval == 0: plotter.refresh()
        if ep % 10 == 0:
            recent = np.mean(plotter.rewards[-20:]) if len(plotter.rewards) >= 20 else np.mean(plotter.rewards)
            print(f"Ep {ep:3d} | Reward: {ep_r:6.1f} | Avg20: {recent:6.1f} | Loss: {avg_l:.4f} | P_init: {p_init:.3f} | Time: {ep_time:.2f}s")
    return plotter.rewards, plotter.p_inits

def compare():
    methods = ['full_vector', 'layer_decoupled', 'node_decoupled']
    results = {}
    for m in methods: results[m] = train(m); plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors, labels = ['blue', 'green', 'red'], ['Full Vector', 'Layer Decoupled', 'Node Decoupled']
    for i, m in enumerate(methods):
        r, p = results[m]
        ma = np.convolve(r, np.ones(20)/20, 'valid')
        axes[0].plot(range(19, len(r)), ma, color=colors[i], label=labels[i])
        axes[1].plot(p, color=colors[i], label=labels[i])
    axes[0].set_title('Reward (MA20)'); axes[0].legend()
    axes[1].set_title('Adaptive P_init'); axes[1].legend()
    plt.tight_layout(); plt.savefig('srrhuif_comparison_v3.png'); plt.show()

if __name__ == "__main__":
    train('node_decoupled')
