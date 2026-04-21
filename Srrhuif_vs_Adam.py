import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import os
import sys
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('Agg')

"""
=========================================================================
SRRHUIF-D3QN vs Adam (Multi-Seed Benchmark Edition)
=========================================================================
- 5개 시드 연속 테스트 자동화
- 순수 로직 검증: 인위적 Gating 배제, 순수 FIR 필터 업데이트만 수행
- [Fix 1] Loss 정상 기록 (MSE)
- [Fix 2] Adam Q-Target 차원 명시화 (dim=1)
=========================================================================
"""

class DualLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg); self.file.write(msg); self.file.flush()
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()

_dual_logger = None
def setup_file_logging(filepath):
    global _dual_logger
    _dual_logger = DualLogger(filepath)
    sys.stdout = _dual_logger

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
JITTER = 1e-12

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    seeds: List[int] = field(default_factory=lambda: [0, 333, 555, 777, 999])
    
    max_episodes: int = 120
    max_steps: int = 500
    batch_size: int = 64
    buffer_size: int = 50000
    warmup_step: int = 100

    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    gamma: float = 0.94
    scale_factor: float = 1.0
    
    # ------ SRRHUIF 파라미터 ---------
    tau_srrhuif: float = 0.02
    update_interval: int = 4
    N_horizon: int = 9
    q_std: float = 5e-4
    r_std: float = 2.0
    
    alpha: float = 0.518
    beta: float = 2.0   
    kappa: float = 0.0
    p_init: float = 0.03
    tikhonov_lambda: float = 0.0
    
    # Adam 파라미터
    adam_lr: float = 3e-4
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 3000
    use_spas: bool = True  
    
    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_std
        self.r_inv = 1.0 / (self.r_std ** 2)
        self.tikhonov_sqrt = float(np.sqrt(self.tikhonov_lambda))
        self.outdir = f"./results_multiseed_v9"
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
parser.add_argument('--warmup_step', type=int, default=cfg.warmup_step)
args, _ = parser.parse_known_args()

cfg.alpha = args.alpha; cfg.beta = args.beta; cfg.r_std = args.r_std
cfg.p_init = args.p_init; cfg.max_episodes = args.episodes
cfg.N_horizon = args.horizon; cfg.batch_size = args.batch
cfg.q_std = args.q_std; cfg.tau_srrhuif = args.tau; cfg.warmup_step = args.warmup_step
cfg.__post_init__()

# =========================================================================
# Shared Network Utils (SRRHUIF)
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

class InputNormalizer:
    def __init__(self, device):
        self.scale = torch.tensor([2.4, 3.0, 0.21, 2.0], dtype=DTYPE, device=device)
    def normalize(self, x):
        if x.dim() == 1: return x / self.scale
        elif x.shape[-1] == len(self.scale): return x / self.scale
        else: return x / self.scale.view(-1, 1)

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
        self.term[idx] = float(done)
        self.count += 1

    @property
    def current_size(self): return min(self.count, self.capacity)

    def sample_batch(self, batch_size: int) -> Dict:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return {'s': self.S[indices].t(), 'a': self.A[indices], 'r': self.R[indices],
                's_next': self.S_next[indices].t(), 'term': self.term[indices]}

# =========================================================================
# SRRHUIF Core Modules
# =========================================================================
class NDCache:
    def __init__(self, info: Dict, cfg: Config, device: str):
        self.layers = {}
        total_forwards = 0
        layer_fwd_slices = []
        for L, nd_layer in enumerate(info['nd_layers']):
            fan_in, fan_out = nd_layer['fan_in'], nd_layer['fan_out']
            n_per = nd_layer['n_per_neuron']
            num_sigma = 2 * n_per + 1
            count = fan_out * num_sigma
            layer_fwd_slices.append((total_forwards, total_forwards + count))
            total_forwards += count
            
            j_idx = torch.arange(fan_out, device=device).view(-1, 1, 1)
            k_idx = torch.arange(fan_in, device=device).view(1, 1, -1)
            w_col_idx = (nd_layer['W_start'] + j_idx * fan_in + k_idx).expand(fan_out, num_sigma, fan_in).contiguous()
            b_col_idx = (nd_layer['b_start'] + j_idx.squeeze(-1)).expand(fan_out, num_sigma).unsqueeze(-1).contiguous()
            
            eye_n_per = torch.eye(n_per, dtype=DTYPE, device=device)
            eye_n_per_batch = eye_n_per.unsqueeze(0).expand(fan_out, -1, -1).clone()
            lamb = cfg.alpha ** 2 * (n_per + cfg.kappa) - n_per
            gamma = float(np.sqrt(n_per + lamb))
            Wm = torch.zeros(2 * n_per + 1, dtype=DTYPE, device=device)
            Wc = torch.zeros(2 * n_per + 1, dtype=DTYPE, device=device)
            Wm[0] = lamb / (n_per + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha ** 2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (n_per + lamb)
            
            self.layers[L] = {
                'w_col_idx': w_col_idx, 'b_col_idx': b_col_idx,
                'eye_n_per': eye_n_per, 'eye_n_per_batch': eye_n_per_batch,
                'Wm_col_f32': Wm.to(DTYPE_FWD).view(1, -1, 1).expand(fan_out, -1, -1).clone(),
                'Wc_f32': Wc.to(DTYPE_FWD), 'gamma': gamma,
                'zero_col_f32': torch.zeros(fan_out, n_per, 1, dtype=DTYPE_FWD, device=device),
                'fan_in': fan_in, 'fan_out': fan_out, 'n_per': n_per, 'num_sigma': num_sigma,
            }
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices

        self.n_per_groups = {}
        for L, nd_layer in enumerate(info['nd_layers']):
            n_per = nd_layer['n_per_neuron']
            if n_per not in self.n_per_groups:
                self.n_per_groups[n_per] = {'layers': [], 'fan_outs': [], 'total_neurons': 0}
            self.n_per_groups[n_per]['layers'].append(L)
            self.n_per_groups[n_per]['fan_outs'].append(nd_layer['fan_out'])
            self.n_per_groups[n_per]['total_neurons'] += nd_layer['fan_out']
        
        for n_per, grp in self.n_per_groups.items():
            grp['eye_grouped'] = torch.eye(n_per, dtype=DTYPE, device=device).unsqueeze(0).expand(grp['total_neurons'], -1, -1).clone()
            grp['S_Q_grouped'] = cfg.q_std * grp['eye_grouped'].clone()
            grp['gamma'] = self.layers[grp['layers'][0]]['gamma']
            offsets = [0]
            for fo in grp['fan_outs']: offsets.append(offsets[-1] + fo)
            grp['offsets'] = offsets

def forward_single(theta, info, x):
    theta = theta.to(DTYPE_FWD); x = x.to(DTYPE_FWD)
    if theta.dim() == 2: theta = theta.squeeze()
    if x.dim() == 1: x = x.unsqueeze(1)
    if x.shape[0] != info['dimS']: x = x.t()
    h = x
    for i in range(info['shared_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        h = F.relu(W @ h + b)
    v = h
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        W = theta[layer['W_start']:layer['W_start'] + layer['W_len']].view(layer['W_shape'])
        b = theta[layer['b_start']:layer['b_start'] + layer['b_len']].view(-1, 1)
        z = W @ v + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = h
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
    v = h
    for i in range(info['shared_end_idx'], info['value_end_idx']):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, v) + b
        v = F.relu(z) if i < info['value_end_idx'] - 1 else z
    a = h
    for i in range(info['value_end_idx'], len(info['layers'])):
        layer = info['layers'][i]
        out_dim, in_dim = layer['W_shape']
        W = thetas[:, layer['W_start']:layer['W_start'] + layer['W_len']].view(num_sigma, out_dim, in_dim)
        b = thetas[:, layer['b_start']:layer['b_start'] + layer['b_len']].view(num_sigma, out_dim, 1)
        z = torch.bmm(W, a) + b
        a = F.relu(z) if i < len(info['layers']) - 1 else z
    return v + (a - a.mean(dim=1, keepdim=True))

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

@torch.no_grad()
def srrhuif_step_nd(theta_current_in, theta_target, neuron_S_info, batch, sp, is_first, nd_cache, q_next_target_cached=None):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    theta_prior = (theta_target if is_first else theta_current_in).clone()
    theta_current = theta_current_in.clone()
    new_S_info_dict = {}

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    unified = nd_cache.unified_thetas
    unified[:] = theta_current.squeeze().to(DTYPE_FWD)
    per_layer = {}

    for L in range(info['num_nd_layers']):
        nd_layer = info['nd_layers'][L]
        lc = nd_cache.layers[L]
        W_start, b_start = nd_layer['W_start'], nd_layer['b_start']

        W_prior = theta_prior.squeeze()[W_start:W_start + nd_layer['W_len']].view(nd_layer['fan_out'], nd_layer['fan_in'])
        b_prior = theta_prior.squeeze()[b_start:b_start + nd_layer['b_len']]
        theta_all_prior_3d = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1).unsqueeze(-1)

        S_3d = neuron_S_info[L]
        if is_first or S_3d is None:
            P_sqrt_prev = np.sqrt(cfg.p_init) * lc['eye_n_per_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_n_per_batch'])

        per_layer[L] = {'theta_all_prior_3d': theta_all_prior_3d, 'P_sqrt_prev': P_sqrt_prev, 'lc': lc, 'fan_in': nd_layer['fan_in'], 'fan_out': nd_layer['fan_out']}

    for n_per_val, grp in nd_cache.n_per_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev'] for L in layers_in_grp], dim=0)

        combined = torch.cat([all_P_sqrt, grp['S_Q_grouped']], dim=2)
        P_sqrt_pred = tria_operation_batch(combined)
        S_pred = safe_inv_tril_batch(P_sqrt_pred, grp['eye_grouped'])
        Y_pred = torch.bmm(S_pred, S_pred.transpose(-2, -1))
        y_pred = torch.bmm(Y_pred, all_theta_3d)
        scaled_P = grp['gamma'] * P_sqrt_pred
        theta_2d = all_theta_3d.squeeze(-1)
        X_sigma_all = torch.cat([theta_2d.unsqueeze(1), theta_2d.unsqueeze(1) + scaled_P.transpose(-2, -1), theta_2d.unsqueeze(1) - scaled_P.transpose(-2, -1)], dim=1)

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            per_layer[L].update({'S_pred': S_pred[s:e], 'Y_pred': Y_pred[s:e], 'y_pred': y_pred[s:e], 'X_sigma_all': X_sigma_all[s:e], 'scaled_P': scaled_P[s:e]})

    for L in range(info['num_nd_layers']):
        pl = per_layer[L]
        X_sigma_f32 = pl['X_sigma_all'].to(DTYPE_FWD)
        fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
        layer_view = unified[fwd_start:fwd_end].view(pl['fan_out'], pl['lc']['num_sigma'], -1)
        layer_view.scatter_(dim=2, index=pl['lc']['w_col_idx'], src=X_sigma_f32[:, :, :pl['fan_in']])
        layer_view.scatter_(dim=2, index=pl['lc']['b_col_idx'], src=X_sigma_f32[:, :, pl['fan_in']:pl['fan_in'] + 1])

    if q_next_target_cached is not None:
        if is_first and cfg.use_spas:
            Q_sigma_f32 = forward_bmm(unified, info, s_next)
            a_best_next = Q_sigma_f32.mean(dim=0).argmax(dim=0)
        else:
            Q_curr = forward_single(theta_current.squeeze(), info, s_next)
            a_best_next = Q_curr.argmax(dim=0)
        q_val_next = q_next_target_cached[a_best_next, torch.arange(batch_sz, device=device)].to(DTYPE)
    
    z_measured = (batch['r'] + cfg.gamma * (1 - batch['term']) * q_val_next).view(-1, 1)
    Q_all_f32 = forward_bmm(unified, info, s_batch)

    for n_per_val, grp in nd_cache.n_per_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']
        all_S_pred = torch.cat([per_layer[L]['S_pred'] for L in layers_in_grp], dim=0)
        all_y_pred = torch.cat([per_layer[L]['y_pred'] for L in layers_in_grp], dim=0)
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        
        all_HT, all_residual = [], []
        for L in layers_in_grp:
            pl = per_layer[L]
            fwd_start, fwd_end = nd_cache.layer_fwd_slices[L]
            Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(pl['fan_out'], pl['lc']['num_sigma'], info['nA'], -1)
            Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
            z_measured_exp = z_measured.unsqueeze(0).expand(pl['fan_out'], -1, -1)
            
            z_hat_f32 = torch.bmm(Z_sigma_T_f32, pl['lc']['Wm_col_f32'])
            Z_dev_f32 = Z_sigma_T_f32 - z_hat_f32
            X_dev_f32 = torch.cat([pl['lc']['zero_col_f32'], pl['scaled_P'].to(DTYPE_FWD), -pl['scaled_P'].to(DTYPE_FWD)], dim=2)
            P_xz_f32 = torch.bmm(X_dev_f32 * pl['lc']['Wc_f32'].view(1, 1, -1), Z_dev_f32.transpose(1, 2))
            
            residual_all = z_measured_exp - z_hat_f32.to(DTYPE)
            HT_all = torch.bmm(pl['Y_pred'], P_xz_f32.to(DTYPE))
            all_HT.append(HT_all); all_residual.append(residual_all)
            
        all_HT = torch.cat(all_HT, dim=0); all_residual = torch.cat(all_residual, dim=0)
        
        combined = torch.cat([all_S_pred, all_HT * cfg.r_inv_sqrt], dim=2)
        S_new_g = tria_operation_batch(combined)
        innov = all_residual + torch.bmm(all_HT.transpose(1, 2), all_theta_3d)
        y_new_g = all_y_pred + torch.bmm(all_HT, cfg.r_inv * innov)
        theta_new_g = robust_solve_spd_batch(S_new_g, y_new_g, grp['eye_grouped'])

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            theta_new_L = theta_new_g[s:e]
            W_new = theta_new_L[:, :per_layer[L]['fan_in'], 0]
            b_new = theta_new_L[:, per_layer[L]['fan_in'], 0]
            nd_layer = info['nd_layers'][L]
            
            theta_flat = theta_current.squeeze()
            theta_flat[nd_layer['W_start']:nd_layer['W_start'] + nd_layer['W_len']] = W_new.reshape(-1)
            theta_flat[nd_layer['b_start']:nd_layer['b_start'] + nd_layer['b_len']] = b_new
            theta_current = theta_flat.view(-1, 1)
            new_S_info_dict[L] = S_new_g[s:e].permute(1, 2, 0)

    new_S_info = [new_S_info_dict[L] for L in range(info['num_nd_layers'])]
    return theta_current, new_S_info

# =========================================================================
# Adam Baseline D3QN Module
# =========================================================================
class AdamD3QN(nn.Module):
    def __init__(self, dimS, nA, cfg):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dimS, cfg.shared_layers[0]), nn.ReLU(),
            nn.Linear(cfg.shared_layers[0], cfg.shared_layers[1]), nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(cfg.shared_layers[1], cfg.value_layers[0]), nn.ReLU(),
            nn.Linear(cfg.value_layers[0], 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(cfg.shared_layers[1], cfg.advantage_layers[0]), nn.ReLU(),
            nn.Linear(cfg.advantage_layers[0], nA)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.randn_(m.weight)
                m.weight.data *= np.sqrt(2.0 / m.in_features)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q

# =========================================================================
# Logger Class
# =========================================================================
class RunLogger:
    def __init__(self):
        self.rewards = []
        self.losses = []
        self.update_times_ms = []
        self.converged_ep = -1

    def add(self, r, l):
        self.rewards.append(r)
        self.losses.append(l)

# =========================================================================
# Training Functions
# =========================================================================
def train_srrhuif_nd(seed: int, is_diag_seed: bool) -> RunLogger:
    set_all_seeds(seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    
    normalizer = InputNormalizer(cfg.device)
    nd_cache = NDCache(info, cfg, cfg.device)
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size, 'normalizer': normalizer, 'device': cfg.device}

    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=cfg.device)
    for layer in info['layers']:
        fan_in, W_len = layer['W_shape'][1], layer['W_len']
        theta[layer['W_start']:layer['W_start'] + W_len] = torch.randn(W_len, dtype=DTYPE, device=cfg.device) * np.sqrt(2.0 / fan_in)
    theta = theta.view(-1, 1)
    theta_target = theta.clone()
    
    neuron_S_info = [1e-6 * nd_cache.layers[L]['eye_n_per'].unsqueeze(-1).expand(-1, -1, info['nd_layers'][L]['fan_out']).clone() 
                     for L in range(info['num_nd_layers'])]

    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)
    batch_hist = deque(maxlen=cfg.N_horizon)
    
    logger = RunLogger()
    steps_done = 0

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=seed + ep)
        ep_r, ep_l = 0, []

        for t in range(cfg.max_steps):
            steps_done += 1
            if steps_done <= cfg.warmup_step: eps = 1.0
            else: eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-(steps_done - cfg.warmup_step) / cfg.eps_decay_steps)
            
            with torch.no_grad():
                s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                q_vals = forward_single(theta.squeeze(), info, normalizer.normalize(s_t_buffer)).squeeze()

            a = env.action_space.sample() if np.random.rand() < eps else q_vals.argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if steps_done > cfg.warmup_step and buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)
                
                # [Fix 1] SRRHUIF의 TD-Loss 연산 및 로깅 (순수 기록용, 업데이트엔 영향 없음)
                with torch.no_grad():
                    s_b = normalizer.normalize(batch['s'].t())
                    ns_b = normalizer.normalize(batch['s_next'].t())
                    Q_tgt_all = forward_single(theta_target.squeeze(), info, ns_b)
                    next_q = Q_tgt_all.max(dim=0)[0]
                    target = batch['r'] + cfg.gamma * (1 - batch['term']) * next_q
                    Q_curr_all = forward_single(theta.squeeze(), info, s_b)
                    q_v = Q_curr_all[batch['a'], torch.arange(cfg.batch_size, device=cfg.device)]
                    ep_l.append(F.mse_loss(q_v, target).item())

                # 순수 SRRHUIF 업데이트 로직
                batch_hist.append(batch)
                if len(batch_hist) == cfg.N_horizon:
                    q_next_caches = []
                    with torch.no_grad():
                        for h in range(cfg.N_horizon):
                            Q_tgt_h = forward_single(theta_target.squeeze(), info, normalizer.normalize(batch_hist[h]['s_next'].t()))
                            q_next_caches.append(Q_tgt_h)

                    for h in range(cfg.N_horizon):
                        theta, neuron_S_info = srrhuif_step_nd(
                            theta, theta_target, neuron_S_info, batch_hist[h], sp,
                            (h == 0), nd_cache, q_next_caches[h])

                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                
                logger.update_times_ms.append((time.perf_counter() - update_start) * 1000)

            if done or trunc: break

        # [Fix 1] Loss 정상 기록
        ep_loss_mean = np.mean(ep_l) if len(ep_l) > 0 else 0.0
        logger.add(ep_r, ep_loss_mean)
        
        if ep >= 20 and np.mean(logger.rewards[-20:]) >= 490.0 and logger.converged_ep == -1:
            logger.converged_ep = ep
            
        if is_diag_seed and ep % 10 == 0:
            print(f"[SRRHUIF|Seed {seed}] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {np.mean(logger.rewards[-20:]):.1f} | Loss: {ep_loss_mean:.4f}")
            
    if logger.converged_ep == -1: logger.converged_ep = cfg.max_episodes
    env.close()
    return logger

def train_adam(seed: int, is_diag_seed: bool) -> RunLogger:
    set_all_seeds(seed)
    env = gym.make(cfg.env_name)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    
    net = AdamD3QN(dimS, nA, cfg).to(cfg.device).to(DTYPE)
    target_net = AdamD3QN(dimS, nA, cfg).to(cfg.device).to(DTYPE)
    target_net.load_state_dict(net.state_dict())
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.adam_lr)
    normalizer = InputNormalizer(cfg.device)
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device)
    
    logger = RunLogger()
    steps_done = 0

    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=seed + ep)
        ep_r, ep_l = 0, []

        for t in range(cfg.max_steps):
            steps_done += 1
            if steps_done <= cfg.warmup_step: eps = 1.0
            else: eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-(steps_done - cfg.warmup_step) / cfg.eps_decay_steps)
            
            with torch.no_grad():
                s_t = normalizer.normalize(torch.as_tensor(s, dtype=DTYPE, device=cfg.device).unsqueeze(0))
                q_vals = net(s_t).squeeze()

            a = env.action_space.sample() if np.random.rand() < eps else q_vals.argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if steps_done > cfg.warmup_step and buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                
                batch = buffer.sample_batch(cfg.batch_size)
                s_b = normalizer.normalize(batch['s'].t())
                ns_b = normalizer.normalize(batch['s_next'].t())
                
                q_v = net(s_b).gather(1, batch['a'].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # [Fix 2] dim=1 로 차원 명확히 고정하여 뷰(View) 충돌 방어
                    next_q = target_net(ns_b).max(dim=1, keepdim=False)[0]
                    target = batch['r'] + cfg.gamma * (1 - batch['term']) * next_q
                
                loss = F.mse_loss(q_v, target)
                ep_l.append(loss.item())  # [Fix 1] Loss 정상 기록
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                for target_param, param in zip(target_net.parameters(), net.parameters()):
                    target_param.data.copy_(cfg.tau_srrhuif * param.data + (1.0 - cfg.tau_srrhuif) * target_param.data)
                
                logger.update_times_ms.append((time.perf_counter() - update_start) * 1000)

            if done or trunc: break

        # [Fix 1] Loss 정상 기록
        ep_loss_mean = np.mean(ep_l) if len(ep_l) > 0 else 0.0
        logger.add(ep_r, ep_loss_mean)
        
        if ep >= 20 and np.mean(logger.rewards[-20:]) >= 490.0 and logger.converged_ep == -1:
            logger.converged_ep = ep
            
        if is_diag_seed and ep % 10 == 0:
            print(f"[Adam   |Seed {seed}] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {np.mean(logger.rewards[-20:]):.1f} | Loss: {ep_loss_mean:.4f}")
            
    if logger.converged_ep == -1: logger.converged_ep = cfg.max_episodes
    env.close()
    return logger

# =========================================================================
# Plotting Functions
# =========================================================================
def plot_multiseed_results(srrhuif_logs, adam_logs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Reward Curve
    ax = axes[0]
    def plot_curve(logs, color, label):
        rewards = np.array([log.rewards for log in logs])
        mean_r = np.mean(rewards, axis=0)
        std_r = np.std(rewards, axis=0)
        episodes = np.arange(1, cfg.max_episodes + 1)
        
        window = 10
        if len(mean_r) >= window:
            ma_mean = np.convolve(mean_r, np.ones(window)/window, mode='valid')
            ma_std = np.convolve(std_r, np.ones(window)/window, mode='valid')
            ep_ma = episodes[window-1:]
            ax.plot(ep_ma, ma_mean, color=color, linewidth=2.5, label=label)
            ax.fill_between(ep_ma, ma_mean - ma_std, ma_mean + ma_std, color=color, alpha=0.2)
        else:
            ax.plot(episodes, mean_r, color=color, linewidth=2.5, label=label)

    plot_curve(srrhuif_logs, 'blue', 'SRRHUIF-ND (2nd-Order)')
    plot_curve(adam_logs, 'red', 'Adam (1st-Order)')
    ax.axhline(500, color='g', linestyle='--', alpha=0.5, label='Max Reward (500)')
    ax.set_title(f'Multi-Seed Reward ({len(cfg.seeds)} Seeds)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.set_xlim(0, cfg.max_episodes); ax.set_ylim(0, 520)
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Efficiency Bar Chart
    ax2 = axes[1]
    s_conv = np.mean([log.converged_ep for log in srrhuif_logs])
    a_conv = np.mean([log.converged_ep for log in adam_logs])
    
    s_ms = np.mean([np.mean(log.update_times_ms) for log in srrhuif_logs])
    a_ms = np.mean([np.mean(log.update_times_ms) for log in adam_logs])

    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width/2, [s_conv, a_conv], width, color=['blue', 'red'], alpha=0.7, label='Episodes to Converge (Lower is Better)')
    ax2.set_ylabel('Episodes to 490+ Avg')
    ax2.set_title('Sample Efficiency vs Computational Cost')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['SRRHUIF-ND', 'Adam'])
    
    # Twin axis for ms/step
    ax3 = ax2.twinx()
    ax3.plot(x, [s_ms, a_ms], 'ko-', markersize=8, linewidth=2, label='Update Time (ms/step)')
    ax3.set_ylabel('Time (ms) per Update Step')
    ax3.set_ylim(0, max(s_ms, a_ms) * 1.5)
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "MultiSeed_Benchmark.png"), dpi=150)
    print(f"\n[✔] 벤치마크 플롯 저장 완료: {cfg.outdir}/MultiSeed_Benchmark.png")

# =========================================================================
# Main Execution
# =========================================================================
def main():
    setup_file_logging(os.path.join(cfg.outdir, "benchmark_log.txt"))
    print(f"============================================================")
    print(f"🚀 SRRHUIF-ND vs Adam Multi-Seed Benchmark (Pure Logic)")
    print(f"Seeds: {cfg.seeds}")
    print(f"Settings: Warmup={cfg.warmup_step}, Batch={cfg.batch_size}, Horizon={cfg.N_horizon}")
    print(f"============================================================\n")

    srrhuif_logs = []
    adam_logs = []

    for seed in cfg.seeds:
        is_diag = (seed == cfg.seeds[0])
        print(f"\n>>> Running Seed {seed} ...")
        
        # 1. SRRHUIF
        start_t = time.time()
        s_log = train_srrhuif_nd(seed, is_diag)
        s_time = time.time() - start_t
        srrhuif_logs.append(s_log)
        
        # 2. Adam
        start_t = time.time()
        a_log = train_adam(seed, is_diag)
        a_time = time.time() - start_t
        adam_logs.append(a_log)
        
        print(f"   [Seed {seed} Done] SRRHUIF: {s_log.converged_ep} eps ({s_time:.1f}s) | Adam: {a_log.converged_ep} eps ({a_time:.1f}s)")

    plot_multiseed_results(srrhuif_logs, adam_logs)

if __name__ == "__main__":
    main()