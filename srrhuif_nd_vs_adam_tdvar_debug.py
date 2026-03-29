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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('Agg')

"""
=========================================================================
SRRHUIF-D3QN v4.1: Adaptive P + Target Variance(T_Var) Debugging
=========================================================================
Changes from v4:
  - Adaptive P_init logic is preserved.
  - ★ TD Target Variance (T_Var) calculation added in SRRHUIF step.
  - ★ Console output updated to include T_Var for monitoring.
=========================================================================
"""

print("=" * 70)
print(f"SRRHUIF-D3QN v4.1 (Adaptive P + T_Var Debug) | PyTorch: {torch.__version__}")
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
JITTER = 1e-14

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
    buffer_size: int = 10000

    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    gamma: float = 0.94
    scale_factor: float = 1.0
    
    # --- SRRHUIF ND params ---
    tau_srrhuif: float = 0.005
    N_horizon: int = 5
    q_std: float = 5e-3
    r_std: float = 1.8

    alpha: float = 0.85
    beta: float = 2.0
    kappa: float = 0.0
    
    # ★ Adaptive P_init variables (Maintained)
    p_init_min: float = 0.001
    p_init_max: float = 0.025
    adaptive_window: int = 15
    use_spas: bool = True   # ★ Sigma Point Action Selection

    # --- Adam D3QN params ---
    tau_adam: float = 0.005
    adam_lr: float = 1e-3

    # --- Exploration ---
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000

    update_interval: int = 1
    use_input_norm: bool = True
    use_compile: bool = False
    plot_interval: int = 10
    seed: int = 0

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_std
        self.r_inv = 1.0 / (self.r_std ** 2)

cfg = Config()

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
    """Pre-compute + Unified FP32 Buffer for ND"""
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

        self.unified_thetas = torch.empty(total_forwards, info['total_params'],
                                          dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards

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
# 3. Forward Functions (FP32 internal)
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
    thetas = thetas.to(DTYPE_FWD)
    x = x.to(DTYPE_FWD)
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


# =========================================================================
# 4. Replay Buffer
# =========================================================================
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


# =========================================================================
# 5. Math Utilities (FP64)
# =========================================================================
def tria_operation_batch(A):
    _, r = torch.linalg.qr(A.transpose(-2, -1).contiguous())
    s = r.transpose(-2, -1).contiguous()
    d = torch.diagonal(s, dim1=-2, dim2=-1)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    s = s * signs.unsqueeze(-1)
    return s

def safe_inv_tril_batch(L_batch, eye_batch):
    result = torch.linalg.solve_triangular(L_batch + JITTER * eye_batch, eye_batch, upper=False)
    return torch.where(torch.isfinite(result), result, eye_batch)

def robust_solve_spd_batch(S_tril_batch, y_batch, eye_batch):
    S_safe = S_tril_batch + JITTER * eye_batch
    z = torch.linalg.solve_triangular(S_safe, y_batch, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.transpose(-2, -1).contiguous(), z, upper=True)
    return torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))


# =========================================================================
# 6. ND Compiled Core Functions
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
    P_xz_f32 = torch.bmm(
        X_dev_f32 * Wc_f32.view(1, 1, -1),
        Z_dev_f32.transpose(1, 2)
    )
    z_hat_f64 = z_hat_f32.to(torch.float64)
    residual_all = z_measured_exp_f64 - z_hat_f64
    HT_all = torch.bmm(Y_pred_f64, P_xz_f32.to(torch.float64))
    return HT_all, residual_all, z_hat_f64


def _nd_meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all,
                          r_inv_sqrt, r_inv, eye_batch):
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
else:
    print("torch.compile disabled or CPU mode.")


# =========================================================================
# 7. Initialize theta
# =========================================================================
def initialize_theta(info, device):
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    for layer in info['layers']:
        fan_in, W_len = layer['W_shape'][1], layer['W_len']
        theta[layer['W_start']:layer['W_start'] + W_len] = \
            torch.randn(W_len, dtype=DTYPE, device=device) * np.sqrt(2.0 / fan_in)
    return theta


# =========================================================================
# 8. ★★★ SRRHUIF ND Step — with SPAS option & T_Var Debug ★★★
# =========================================================================
@torch.no_grad()
def srrhuif_step_nd(theta_current_in, theta_target, neuron_S_info, batch, sp,
                    is_first, p_init_val, nd_cache):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info, total_loss, layer_count = [], 0.0, 0

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    # =====================================================================
    # Phase 1a: Time Update (FP64) + Scatter to FP32 unified buffer
    # =====================================================================
    unified = nd_cache.unified_thetas
    unified[:] = theta_current.squeeze().to(DTYPE_FWD)

    time_data = []
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
            P_sqrt_prev = np.sqrt(p_init_val) * lc['eye_n_per_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_n_per_batch'])

        S_pred, Y_pred, y_pred, X_sigma_all, scaled_P = _nd_time_update_core(
            theta_all_prior_3d, P_sqrt_prev, lc['S_Q_cached'], lc['eye_n_per_batch'], lc['gamma'])

        X_sigma_f32 = X_sigma_all.to(DTYPE_FWD)
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :fan_in])
        layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, fan_in:fan_in + 1])

        time_data.append({
            'nd_layer': nd_layer, 'lc': lc,
            'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per,
            'W_start': W_start, 'b_start': b_start,
            'theta_all_prior': theta_all_prior, 'theta_all_prior_3d': theta_all_prior_3d,
            'S_pred': S_pred, 'Y_pred': Y_pred, 'y_pred': y_pred, 'scaled_P': scaled_P,
        })

    # =====================================================================
    # DDQN Target
    # =====================================================================
    if is_first:
        if cfg.use_spas:
            Q_sigma_f32 = forward_bmm(unified, info, s_next)
            Q_sigma_mean = Q_sigma_f32.mean(dim=0)
            a_best_next = Q_sigma_mean.argmax(dim=0)
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

    # 측정치 (TD Target)
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)

    # ★ 배치 내 타겟의 분산(Variance) 계산 (디버깅용)
    target_var = torch.var(z_measured).item()

    # =====================================================================
    # Phase 1b: Single forward_bmm for measurement (FP32)
    # =====================================================================
    Q_all_f32 = forward_bmm(unified, info, s_batch)

    # =====================================================================
    # Phase 1c: HT Computation — Mixed Precision Core
    # =====================================================================
    layer_data = []
    for L in range(info['num_nd_layers']):
        td = time_data[L]
        lc = td['lc']
        fan_out = td['fan_out']

        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(fan_out, lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)

        z_measured_exp = z_measured.unsqueeze(0).expand(fan_out, -1, -1)

        HT_all, residual_all, z_hat_f64 = _nd_compute_ht_core(
            Z_sigma_T_f32, lc['Wm_col_f32'], lc['Wc_f32'], lc['zero_col_f32'],
            td['scaled_P'].to(DTYPE_FWD), z_measured_exp, td['Y_pred'])

        loss_L = torch.mean(residual_all ** 2)
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
            cfg.r_inv_sqrt, cfg.r_inv, data['lc']['eye_n_per_batch'])

        invalid = ~torch.isfinite(theta_new_all).all(dim=(1, 2))
        if invalid.any():
            theta_new_all[invalid] = data['theta_all_prior'][invalid].unsqueeze(-1)

        W_new = theta_new_all[:, :data['fan_in'], 0]
        b_new = theta_new_all[:, data['fan_in'], 0]
        theta_flat = theta_current.squeeze()
        theta_flat[data['W_start']:data['W_start'] + data['nd_layer']['W_len']] = W_new.reshape(-1)
        theta_flat[data['b_start']:data['b_start'] + data['nd_layer']['b_len']] = b_new
        theta_current = theta_flat.view(-1, 1)

        new_S_info.append(S_new_all.permute(1, 2, 0))
        total_loss = torch.tensor(0.0, dtype=DTYPE, device=device)
        total_loss = total_loss + data['loss']

    # ★ 리턴값에 target_var 추가
    return theta_current, new_S_info, (total_loss / layer_count).item(), target_var


# =========================================================================
# 9. Adam D3QN Module
# =========================================================================
class D3QNModule(nn.Module):
    """Standard PyTorch D3QN (Dueling Double DQN) for Adam training"""
    def __init__(self, dimS, nA, config):
        super().__init__()
        layers = []
        sizes = [dimS] + config.shared_layers
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)

        v_sizes = [config.shared_layers[-1]] + config.value_layers + [1]
        v_layers = []
        for i in range(len(v_sizes) - 1):
            v_layers.append(nn.Linear(v_sizes[i], v_sizes[i + 1]))
            if i < len(v_sizes) - 2: v_layers.append(nn.ReLU())
        self.value_head = nn.Sequential(*v_layers)

        a_sizes = [config.shared_layers[-1]] + config.advantage_layers + [nA]
        a_layers = []
        for i in range(len(a_sizes) - 1):
            a_layers.append(nn.Linear(a_sizes[i], a_sizes[i + 1]))
            if i < len(a_sizes) - 2: a_layers.append(nn.ReLU())
        self.advantage_head = nn.Sequential(*a_layers)

    def forward(self, x):
        h = self.shared(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


# =========================================================================
# 10. Training Loops
# =========================================================================
class LivePlotter:
    """Live plotting during training + data logging"""
    def __init__(self, method_name: str, max_episodes: int):
        self.method_name = method_name
        self.rewards, self.losses, self.p_inits = [], [], []
        self.total_time = 0.0
        self.avg_step_time = 0.0
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 4))
        
        # Reward plot
        self.ax_r = self.axes[0]
        self.line_r_raw, = self.ax_r.plot([], [], 'b-', alpha=0.3)
        self.line_r_ma, = self.ax_r.plot([], [], 'b-', linewidth=2)
        self.ax_r.axhline(y=195, color='g', linestyle='--', alpha=0.5)
        self.ax_r.set_xlim(0, max_episodes); self.ax_r.set_ylim(0, 520)
        self.ax_r.set_title(f'Reward ({method_name})')
        
        # Loss plot
        self.ax_l = self.axes[1]
        self.line_l, = self.ax_l.plot([], [], 'r-', linewidth=1.5)
        self.ax_l.set_title('TD Loss')
        
        # P_init plot
        self.ax_p = self.axes[2]
        self.line_p, = self.ax_p.plot([], [], 'g-', linewidth=2)
        self.ax_p.set_title('Adaptive P_init' if 'SRRHUIF' in method_name else 'Info')
        if 'SRRHUIF' in method_name:
            self.ax_p.set_ylim(0, cfg.p_init_max * 1.2)
        
        plt.tight_layout()
        self.filename = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    def add(self, reward, loss, p_init=0.0):
        self.rewards.append(reward)
        self.losses.append(max(loss, 1e-10))
        self.p_inits.append(p_init)
    
    def refresh(self):
        """Save current plot to file"""
        ep_range = range(len(self.rewards))
        self.line_r_raw.set_data(ep_range, self.rewards)
        if len(self.rewards) >= 20:
            ma = np.convolve(self.rewards, np.ones(20)/20, 'valid')
            self.line_r_ma.set_data(range(19, len(self.rewards)), ma)
        self.line_l.set_data(ep_range, self.losses)
        self.line_p.set_data(ep_range, self.p_inits)
        for ax in self.axes:
            ax.relim(); ax.autoscale_view()
        self.axes[0].set_ylim(0, max(max(self.rewards, default=520) * 1.1, 520))
        plt.savefig(f'{self.filename}_live.png', dpi=100)
    
    def close(self):
        plt.close(self.fig)


def train_srrhuif_nd():
    """Train SRRHUIF Node Decoupled D3QN"""
    set_all_seeds(cfg.seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    spas_str = "SPAS" if cfg.use_spas else "StdDQN"
    print(f"\n{'='*60}")
    print(f"Training SRRHUIF-ND ({spas_str}) | Params: {info['total_params']} "
          f"| R={cfg.r_std} | τ={cfg.tau_srrhuif}")
    print(f"{'='*60}")

    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    nd_cache = NDCache(info, cfg, cfg.device)
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size,
          'normalizer': normalizer, 'device': cfg.device}

    theta = initialize_theta(info, cfg.device).view(-1, 1)
    theta_target = theta.clone()

    neuron_S_info = [
        1e-6 * nd_cache.get(L)['eye_n_per'].unsqueeze(-1).expand(
            -1, -1, nd['fan_out']).clone()
        for L, nd in enumerate(info['nd_layers'])
    ]

    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)
    batch_hist = deque(maxlen=cfg.N_horizon)
    logger = LivePlotter(f"SRRHUIF-ND ({spas_str})", cfg.max_episodes)
    steps_done = 0

    train_start_time = time.time()
    update_times = []

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        
        # ★ ep_var 리스트 추가 (에피소드 내의 T_Var 값들을 모으기 위해)
        ep_r, ep_l, ep_var, ep_start = 0, [], [], time.time()
        
        start_idx = max(0, len(logger.rewards) - cfg.adaptive_window)
        current_score = np.mean(logger.rewards[start_idx:]) if logger.rewards else 0
        gap = max(0.0, min(1.0, 1.0 - current_score / cfg.max_steps))
        
        # Adaptive P_init 로직 유지
        p_init = cfg.p_init_min + (cfg.p_init_max - cfg.p_init_min) * gap

        for t in range(cfg.max_steps):
            steps_done += 1
            eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-steps_done / cfg.eps_decay_steps)
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                    s_t = s_t_buffer
                    if normalizer: s_t = normalizer.normalize(s_t)
                    a = forward_single(theta.squeeze(), info, s_t).squeeze().argmax().item()

            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()

                batch = buffer.sample_batch(cfg.batch_size)
                batch_hist.append(batch)
                if len(batch_hist) == cfg.N_horizon:
                    for h in range(cfg.N_horizon):
                        is_first = (h == 0)
                        
                        # ★ 리턴값 4개를 모두 받도록 수정 (t_var 추가)
                        theta, neuron_S_info, l_val, t_var = srrhuif_step_nd(
                            theta, theta_target, neuron_S_info, batch_hist[h],
                            sp, is_first, p_init, nd_cache)
                        
                        ep_l.append(l_val)
                        ep_var.append(t_var)  # ★ T_Var 수집
                        
                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta

                update_times.append(time.perf_counter() - update_start) 

            if done or trunc: break

        avg_l = np.mean(ep_l) if ep_l else 0
        avg_v = np.mean(ep_var) if ep_var else 0  # ★ 에피소드 평균 T_Var 계산
        
        logger.add(ep_r, avg_l, p_init)
        
        if ep % cfg.plot_interval == 0:
            logger.refresh()
            
        if ep % 5 == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            
            # ★ 터미널 출력에 T_Var 추가
            print(f"[SRRHUIF] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} "
                  f"| Loss: {avg_l:.4f} | T_Var: {avg_v:.4f} | P_init: {p_init:.4f} | Time: {time.time()-ep_start:.2f}s")
            
    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
    env.close()
    logger.refresh()
    logger.close()
    return logger


def train_adam():
    """Train Adam-based D3QN"""
    set_all_seeds(cfg.seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    print(f"\n{'='*60}")
    print(f"Training Adam D3QN | lr={cfg.adam_lr} | τ={cfg.tau_adam}")
    print(f"{'='*60}")

    device = cfg.device
    policy_net = D3QNModule(dimS, nA, cfg).to(device).double()
    target_net = D3QNModule(dimS, nA, cfg).to(device).double()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.adam_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    normalizer = InputNormalizer(device) if cfg.use_input_norm else None
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, device)
    logger = LivePlotter("Adam D3QN", cfg.max_episodes)
    steps_done = 0

    train_start_time = time.time()
    update_times = []

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_r, ep_l, ep_start = 0, [], time.time()

        for t in range(cfg.max_steps):
            steps_done += 1
            eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-steps_done / cfg.eps_decay_steps)
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(s, dtype=DTYPE, device=device)
                    if normalizer: s_t = normalizer.normalize(s_t)
                    a = policy_net(s_t.unsqueeze(0)).argmax(dim=1).item()

            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done or trunc)
            s, ep_r = ns, ep_r + r

            if buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)
                s_b = batch['s'].t()
                s_next_b = batch['s_next'].t()
                if normalizer:
                    s_b = normalizer.normalize(s_b)
                    s_next_b = normalizer.normalize(s_next_b)

                q_vals = policy_net(s_b).gather(1, batch['a'].unsqueeze(1)).squeeze()

                with torch.no_grad():
                    a_best = policy_net(s_next_b).argmax(dim=1)
                    q_next = target_net(s_next_b).gather(1, a_best.unsqueeze(1)).squeeze()
                    q_target = batch['r'] + cfg.gamma * (1 - batch['term']) * q_next

                loss = F.mse_loss(q_vals, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                ep_l.append(loss.item())

                # Soft update
                with torch.no_grad():
                    for p, t_p in zip(policy_net.parameters(), target_net.parameters()):
                        t_p.data.copy_(cfg.tau_adam * p.data + (1 - cfg.tau_adam) * t_p.data)

            if done or trunc: break
        
        logger.total_time = time.time() - train_start_time
        logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
        scheduler.step()
        avg_l = np.mean(ep_l) if ep_l else 0
        logger.add(ep_r, avg_l)
        if ep % cfg.plot_interval == 0:
            logger.refresh()
        if ep % 10 == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            print(f"[Adam]    Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} "
                  f"| Loss: {avg_l:.4f} | Time: {time.time()-ep_start:.2f}s")

    env.close()
    logger.refresh()  # final save
    logger.close()
    return logger


# =========================================================================
# 11. Comparison Plot
# =========================================================================
def plot_comparison(srrhuif_log: LivePlotter, adam_log: LivePlotter):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ma_window = 20
    spas_str = "SPAS" if cfg.use_spas else "StdDQN"
    fig.suptitle(f"SRRHUIF-ND ({spas_str}) vs Adam D3QN | "
                 f"R={cfg.r_std} Q={cfg.q_std} γ={cfg.gamma}", fontsize=14)

    # Row 1: Rewards
    ax = axes[0]
    ax.plot(srrhuif_log.rewards, 'b-', alpha=0.2, label='SRRHUIF raw')
    ax.plot(adam_log.rewards, 'r-', alpha=0.2, label='Adam raw')
    if len(srrhuif_log.rewards) >= ma_window:
        ma_s = np.convolve(srrhuif_log.rewards, np.ones(ma_window) / ma_window, 'valid')
        ax.plot(range(ma_window - 1, len(srrhuif_log.rewards)), ma_s, 'b-', linewidth=2, label=f'SRRHUIF MA{ma_window}')
    if len(adam_log.rewards) >= ma_window:
        ma_a = np.convolve(adam_log.rewards, np.ones(ma_window) / ma_window, 'valid')
        ax.plot(range(ma_window - 1, len(adam_log.rewards)), ma_a, 'r-', linewidth=2, label=f'Adam MA{ma_window}')
    ax.axhline(y=195, color='g', linestyle='--', alpha=0.5)
    ax.set_title('Episode Reward'); ax.legend(fontsize=8)

    # Row 1: Loss
    ax = axes[1]
    ax.plot(srrhuif_log.losses, 'b-', alpha=0.7, label='SRRHUIF')
    ax.plot(adam_log.losses, 'r-', alpha=0.7, label='Adam')
    ax.set_title('TD Loss'); ax.legend()
    
    # Row 2: Summary stats
    ax = axes[2]
    ax.axis('off')
    last50_s = np.mean(srrhuif_log.rewards[-50:]) if len(srrhuif_log.rewards) >= 50 else np.mean(srrhuif_log.rewards)
    last50_a = np.mean(adam_log.rewards[-50:]) if len(adam_log.rewards) >= 50 else np.mean(adam_log.rewards)
    max_s = max(srrhuif_log.rewards)
    max_a = max(adam_log.rewards)
    first_195_s = next((i for i, r in enumerate(srrhuif_log.rewards) if r >= 195), -1)
    first_195_a = next((i for i, r in enumerate(adam_log.rewards) if r >= 195), -1)

    def first_ma_above(rewards, threshold, window=20):
        if len(rewards) < window: return -1
        ma = np.convolve(rewards, np.ones(window)/window, 'valid')
        idx = next((i for i, v in enumerate(ma) if v >= threshold), -1)
        return idx + window - 1 if idx >= 0 else -1

    first_400_s = first_ma_above(srrhuif_log.rewards, 400)
    first_400_a = first_ma_above(adam_log.rewards, 400)

    summary = (
        f"{'Metric':<25} {'SRRHUIF':>10} {'Adam':>10}\n"
        f"{'─' * 47}\n"
        f"{'Last 50 avg':<25} {last50_s:>10.1f} {last50_a:>10.1f}\n"
        f"{'Max reward':<25} {max_s:>10.1f} {max_a:>10.1f}\n"
        f"{'First ep ≥195':<25} {first_195_s:>10d} {first_195_a:>10d}\n"
        f"{'First MA20 ≥400':<25} {first_400_s:>10d} {first_400_a:>10d}\n"
        f"{'Total Time (s)':<25} {srrhuif_log.total_time:>10.1f} {adam_log.total_time:>10.1f}\n"
        f"{'Avg Step Time (ms)':<25} {srrhuif_log.avg_step_time:>10.2f} {adam_log.avg_step_time:>10.2f}\n"
        f"{'─' * 47}\n"
        f"SPAS: {'ON' if cfg.use_spas else 'OFF'}\n"
        f"Params: R={cfg.r_std} Q={cfg.q_std} γ={cfg.gamma}\n"
        f"Horizon={cfg.N_horizon} Batch={cfg.batch_size}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    filename = f"compare_SRRHUIF-ND_{spas_str}_vs_Adam_R{cfg.r_std}_Q{cfg.q_std}_g{cfg.gamma}.png"
    plt.savefig(filename, dpi=150)
    print(f"\nComparison plot saved: {filename}")
    plt.close()


# =========================================================================
# 12. Main
# =========================================================================
def main():
    print(f"\n{'#' * 70}")
    print(f"  SRRHUIF-ND ({'SPAS' if cfg.use_spas else 'Standard'}) vs Adam D3QN Comparison")
    print(f"  Env: {cfg.env_name} | Episodes: {cfg.max_episodes} | Seed: {cfg.seed}")
    print(f"{'#' * 70}")

    srrhuif_log = train_srrhuif_nd()
    plt.close('all')

    adam_log = train_adam()
    plt.close('all')

    plot_comparison(srrhuif_log, adam_log)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()