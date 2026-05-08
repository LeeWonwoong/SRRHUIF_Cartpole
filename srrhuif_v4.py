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
import sys
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('Agg')

# =========================================================================
# Dual Output Logger (console + file) & File-only Print
# =========================================================================
class DualLogger:
    """콘솔 + 파일 동시 출력"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()
        
    def write_file_only(self, msg):
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

_dual_logger = None

def setup_file_logging(filepath):
    global _dual_logger
    _dual_logger = DualLogger(filepath)
    sys.stdout = _dual_logger
    print(f"[Logging] 모든 stdout이 저장됨: {filepath}")

def close_file_logging():
    global _dual_logger
    if _dual_logger is not None:
        sys.stdout = _dual_logger.stdout
        _dual_logger.close()
        _dual_logger = None

def file_print(*args, **kwargs):
    """터미널에는 출력하지 않고 텍스트 파일(training_log.txt)에만 기록합니다."""
    global _dual_logger
    if _dual_logger is not None:
        msg = " ".join(map(str, args))
        _dual_logger.write_file_only(msg + "\n")

print("=" * 70)
print(f"SRRHUIF-DDQN/D3QN v12.0 (Full Diagnostics | Node/Layer/FV Modes) | PyTorch: {torch.__version__}")
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
JITTER = 1e-7
JITTER_TRIA = 1e-7
# =========================================================================
# 1. Configuration
# =========================================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_episodes: int = 120
    max_steps: int = 500
    batch_size: int = 32
    buffer_size: int = 50000

    # Decoupling Mode 
    #   'node'  = per-neuron block-diagonal (mean-field, 가장 가벼움)
    #   'layer' = per-layer joint (K-FAC-like, within-layer covariance 보존)
    #   'fv'    = full vector (모든 파라미터를 한 블록으로, 가장 정확하지만 무거움)
    decoupling_mode: str = 'fv'
    
    # [FIR 철학] h=0의 prior source
    #   'target' = target net (현재 코드 기존 동작)
    #   'init'   = 학습 시작시 frozen된 θ_init (RHE/FIR 정신에 더 가까움)
    h0_prior_source: str = 'init'
    
    # 초기화 방식 (he / orthogonal)
    init_scheme: str = 'he'

    shared_layers: List[int] = field(default_factory=lambda: [8, 8])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    q_layers: List[int] = field(default_factory=lambda: [8])

    use_dueling: bool = False  # False로 두어 순수 DDQN 아키텍처 사용 (Layer 모드 최적화)
    
    gamma: float = 0.94
    scale_factor: float = 1.0
    
    tau_srrhuif: float = 0.02
    N_horizon: int = 10
    q_init: float = 1e-2
    q_end: float = 1e-2

    r_init: float = 3
    r_end: float = 3
    huber_c: float = 3.0        
    
    # 🎯 NEW: N-step 옵션 추가
    use_n_step: bool = True
    n_step_size: int = 4

    # P_init at h=0 (FIR 비정보적 prior 흉내).
    p_init: float = 0.3
    
    alpha: float = 0.99
    beta: float = 2.0   
    kappa: float = 0.0
    max_k_gain: float = 0.0
    
    use_spas: bool = False  

    max_layer_step: float = 0.0
    
    # [FERRARI SETTINGS] Layer + Orthogonal Init 세팅
    tikhonov_lambda: float = 1e-8

    eps_start: float = 0.99
    eps_end: float = 0.01
    eps_decay_steps: int = 2000

    warmup_step : int = 0
    update_interval: int = 4
    use_input_norm: bool = True
    use_compile: bool = True
    plot_interval: int = 60
    log_interval : int = 1
    
    seed: int = 0
    network_seed: Optional[int] = 0
    env_seed: Optional[int] = 0
    
    use_full_eigvalsh: bool = True
    diag_ref_states: bool = True
    diag_argmax_flip: bool = True
    diag_eff_rank: bool = True
    diag_horizon_cond: bool = True
    diag_buffer: bool = True         
    save_file_log: bool = True       

    def __post_init__(self):
        # ── 옵션 값 검증 (Config에서 직접 편집했을 때 오타 잡기) ──
        valid_modes = {'node', 'layer', 'fv'}
        if self.decoupling_mode not in valid_modes:
            raise ValueError(
                f"decoupling_mode='{self.decoupling_mode}' invalid. "
                f"Must be one of {valid_modes}."
            )
        valid_h0 = {'target', 'init'}
        if self.h0_prior_source not in valid_h0:
            raise ValueError(
                f"h0_prior_source='{self.h0_prior_source}' invalid. "
                f"Must be one of {valid_h0}."
            )
        valid_init = {'orthogonal', 'he'}
        if self.init_scheme not in valid_init:
            raise ValueError(
                f"init_scheme='{self.init_scheme}' invalid. "
                f"Must be one of {valid_init}."
            )
        
        self.r_inv_sqrt = 1.0 / self.r_init
        self.r_inv = 1.0 / (self.r_init ** 2)
        duel_str = "D3QN" if self.use_dueling else "DDQN"
        self.param_str = (
            f"mode_{self.decoupling_mode}_{duel_str}_init_{self.init_scheme}_"
            f"h0prior_{self.h0_prior_source}_a{self.alpha}_r{self.r_init}_"
            f"p{self.p_init}_batch{self.batch_size}_netseed{self.network_seed}"
        )
        self.outdir = f"./results_cartpole/{self.param_str}"
        os.makedirs(self.outdir, exist_ok=True)

cfg = Config()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default=cfg.decoupling_mode, choices=['node', 'layer', 'fv'],
                    help="'node' = per-neuron, 'layer' = per-layer joint (K-FAC-like), 'fv' = full vector")
parser.add_argument('--h0_prior', type=str, default=cfg.h0_prior_source,
                    choices=['target', 'init'],
                    help="h=0 prior source: 'target' (target net) or 'init' (frozen θ_init, FIR philosophy)")
parser.add_argument('--init_scheme', type=str, default=cfg.init_scheme,
                    choices=['orthogonal', 'he'])
parser.add_argument('--dueling', action='store_true', default=cfg.use_dueling)
parser.add_argument('--alpha', type=float, default=cfg.alpha)
parser.add_argument('--beta', type=float, default=cfg.beta)
parser.add_argument('--r_init', type=float, default=cfg.r_init)
parser.add_argument('--p_init', type=float, default=cfg.p_init)
parser.add_argument('--episodes', type=int, default=cfg.max_episodes)
parser.add_argument('--batch', type=int, default=cfg.batch_size)
parser.add_argument('--tau', type=float, default=cfg.tau_srrhuif)
parser.add_argument('--tikhonov', type=float, default=cfg.tikhonov_lambda)
parser.add_argument('--seed', type=int, default=cfg.seed)
parser.add_argument('--network_seed', type=int, default=cfg.network_seed)
parser.add_argument('--env_seed', type=int, default=cfg.env_seed)
args, _ = parser.parse_known_args()

cfg.decoupling_mode = args.mode
cfg.h0_prior_source = args.h0_prior
cfg.init_scheme = args.init_scheme
cfg.use_dueling = args.dueling
cfg.alpha = args.alpha
cfg.beta = args.beta
cfg.r_init = args.r_init
cfg.p_init = args.p_init
cfg.max_episodes = args.episodes
cfg.batch_size = args.batch
cfg.tau_srrhuif = args.tau
cfg.tikhonov_lambda = args.tikhonov
cfg.seed = args.seed
cfg.network_seed = args.network_seed
cfg.env_seed = args.env_seed
cfg.__post_init__()

# =========================================================================
# 2. Network Info & Unified Cache
# =========================================================================
def create_network_info(dimS: int, nA: int, config: Config) -> Dict:
    info = {'dimS': dimS, 'nA': nA, 'layers': [], 'filter_layers': [], 'use_dueling': config.use_dueling}
    idx, ld_idx = 0, 0
    def add_layers(sizes, type_str):
        nonlocal idx, ld_idx
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W_len = fan_out * fan_in
            b_len = fan_out
            param_len = W_len + b_len
            
            # [핵심 추상화] Node vs Layer vs FV 블록 크기 분기
            if config.decoupling_mode == 'node':
                block_size = fan_in + 1
                num_blocks = fan_out
            elif config.decoupling_mode == 'layer':
                block_size = param_len
                num_blocks = 1
            else:  # 'fv' - filter_layers는 안 쓰지만 forward용 layers 정보는 필요
                block_size = param_len  # placeholder (FV에선 사용 안 함)
                num_blocks = 1

            layer = {
                'type': type_str, 'layer_idx': i,
                'W_start': idx, 'W_len': W_len, 'W_shape': (fan_out, fan_in),
                'b_start': idx + W_len, 'b_len': b_len,
                'fan_in': fan_in, 'fan_out': fan_out,
            }
            idx += param_len
            info['layers'].append(layer)
            # FV 모드는 filter_layers 안 채움 (FilterCacheFV가 별도 처리)
            if config.decoupling_mode != 'fv':
                info['filter_layers'].append({
                    'global_idx': ld_idx, 'type': type_str, 'local_idx': i,
                    'fan_in': fan_in, 'fan_out': fan_out, 
                    'block_size': block_size, 'num_blocks': num_blocks, 'param_len': param_len,
                    'W_start': layer['W_start'], 'W_len': layer['W_len'],
                    'b_start': layer['b_start'], 'b_len': layer['b_len']})
                ld_idx += 1
            
    shared_out = config.shared_layers[-1]
    add_layers([dimS] + config.shared_layers, 'shared')
    info['shared_end_idx'] = len(info['layers'])
    
    if config.use_dueling:
        add_layers([shared_out] + config.value_layers + [1], 'value')
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + config.advantage_layers + [nA], 'advantage')
    else:
        info['value_end_idx'] = len(info['layers'])
        add_layers([shared_out] + config.q_layers + [nA], 'q_layer')
        
    info['total_params'] = idx
    info['num_filter_layers'] = len(info['filter_layers'])
    return info

class FilterCache:
    def __init__(self, info: Dict, cfg: Config, device: str):
        self.layers = {}
        total_forwards = 0
        layer_fwd_slices = []
        
        for L, fl in enumerate(info['filter_layers']):
            block_size = fl['block_size']
            num_blocks = fl['num_blocks']
            num_sigma = 2 * block_size + 1
            count = num_blocks * num_sigma
            layer_fwd_slices.append((total_forwards, total_forwards + count))
            total_forwards += count
            
            lamb = cfg.alpha ** 2 * (block_size + cfg.kappa) - block_size
            gamma = float(np.sqrt(block_size + lamb))
            Wm = torch.zeros(num_sigma, dtype=DTYPE, device=device)
            Wc = torch.zeros(num_sigma, dtype=DTYPE, device=device)
            Wm[0] = lamb / (block_size + lamb)
            Wc[0] = Wm[0] + (1 - cfg.alpha ** 2 + cfg.beta)
            Wm[1:] = Wc[1:] = 0.5 / (block_size + lamb)
            
            eye_block = torch.eye(block_size, dtype=DTYPE, device=device)
            eye_block_batch = eye_block.unsqueeze(0).expand(num_blocks, -1, -1).clone()
            S_Q_cached = cfg.q_init * eye_block_batch.clone()
            
            Wm_col_f32 = Wm.to(DTYPE_FWD).view(1, -1, 1).expand(num_blocks, -1, -1).clone()
            Wc_f32 = Wc.to(DTYPE_FWD)
            zero_col_f32 = torch.zeros(num_blocks, block_size, 1, dtype=DTYPE_FWD, device=device)
            
            layer_dict = {
                'eye_block': eye_block, 'eye_block_batch': eye_block_batch,
                'Wm': Wm, 'Wc': Wc, 'gamma': gamma,
                'block_size': block_size, 'num_blocks': num_blocks, 'num_sigma': num_sigma,
                'S_Q_cached': S_Q_cached,
                'Wm_col_f32': Wm_col_f32, 'Wc_f32': Wc_f32, 'zero_col_f32': zero_col_f32,
            }
            
            # Node 모드일 때만 흩뿌리기용 인덱스 필요 (Layer 모드는 그냥 연속 메모리 카피)
            if cfg.decoupling_mode == 'node':
                j_idx = torch.arange(fl['fan_out'], device=device).view(-1, 1, 1)
                k_idx = torch.arange(fl['fan_in'], device=device).view(1, 1, -1)
                layer_dict['w_col_idx'] = (fl['W_start'] + j_idx * fl['fan_in'] + k_idx).expand(fl['fan_out'], num_sigma, fl['fan_in']).contiguous()
                layer_dict['b_col_idx'] = (fl['b_start'] + j_idx.squeeze(-1)).expand(fl['fan_out'], num_sigma).unsqueeze(-1).contiguous()
                
            self.layers[L] = layer_dict
            
        self.unified_thetas = torch.empty(total_forwards, info['total_params'], dtype=DTYPE_FWD, device=device)
        self.layer_fwd_slices = layer_fwd_slices
        self.total_forwards = total_forwards

        # 연산 최적화를 위해 block_size가 같은 층들끼리 묶기
        self.block_groups = {}
        for L, fl in enumerate(info['filter_layers']):
            bs = fl['block_size']
            if bs not in self.block_groups:
                self.block_groups[bs] = {'layers': [], 'num_blocks_list': [], 'total_blocks': 0}
            grp = self.block_groups[bs]
            grp['layers'].append(L)
            grp['num_blocks_list'].append(fl['num_blocks'])
            grp['total_blocks'] += fl['num_blocks']
        
        for bs, grp in self.block_groups.items():
            total_b = grp['total_blocks']
            grp['eye_grouped'] = torch.eye(bs, dtype=DTYPE, device=device).unsqueeze(0).expand(total_b, -1, -1).clone()
            grp['gamma'] = self.layers[grp['layers'][0]]['gamma']
            offsets = [0]
            for nb in grp['num_blocks_list']:
                offsets.append(offsets[-1] + nb)
            grp['offsets'] = offsets

    def get(self, layer_idx: int) -> Dict:
        return self.layers[layer_idx]

class FilterCacheFV:
    """Full Vector mode용 캐시. 전체 θ ∈ R^n_x를 하나의 블록으로."""
    def __init__(self, info: Dict, cfg: Config, device: str):
        n_x = info['total_params']
        self.n_x = n_x
        self.num_sigma = 2 * n_x + 1
        
        # UKF weights
        lam = cfg.alpha**2 * (n_x + cfg.kappa) - n_x
        self.gamma_sigma = float(np.sqrt(n_x + lam))
        Wm = np.full(self.num_sigma, 0.5 / (n_x + lam))
        Wc = Wm.copy()
        Wm[0] = lam / (n_x + lam)
        Wc[0] = Wm[0] + (1 - cfg.alpha**2 + cfg.beta)
        self.Wm = torch.tensor(Wm, dtype=DTYPE, device=device)  # [num_sigma]
        self.Wc = torch.tensor(Wc, dtype=DTYPE, device=device)
        
        # 자주 쓰는 buffer
        self.eye_n = torch.eye(n_x, dtype=DTYPE, device=device)
        # forward용 tensor (sigma points × n_x)
        self.unified_thetas = torch.empty(self.num_sigma, n_x, dtype=DTYPE_FWD, device=device)

class InputNormalizer:
    def __init__(self, device):
        self.scale = torch.tensor([2.4, 3.0, 0.21, 2.0], dtype=DTYPE, device=device)
    def normalize(self, x):
        if x.dim() == 1: return x / self.scale
        elif x.shape[-1] == len(self.scale): return x / self.scale
        else: return x / self.scale.view(-1, 1)

# =========================================================================
# 3. Forward Functions & Replay Buffer
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

    if info['use_dueling']:
        return (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    else:
        return a.to(DTYPE)

def forward_single_with_shared(theta, info, x):
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

    if info['use_dueling']:
        Q = (v + (a - a.mean(dim=0, keepdim=True))).to(DTYPE)
    else:
        Q = a.to(DTYPE)
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

    if info['use_dueling']:
        return (v + (a - a.mean(dim=1, keepdim=True))).to(DTYPE)
    else:
        return a.to(DTYPE)

class TensorReplayBuffer:
    def __init__(self, capacity: int, dimS: int, device: str, cfg: Config): # 👈 cfg 인자 추가
        self.capacity, self.count, self.device = capacity, 0, device
        self.S = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.A = torch.zeros(capacity, dtype=torch.long, device=device)
        self.R = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.S_next = torch.zeros(capacity, dimS, dtype=DTYPE, device=device)
        self.term = torch.zeros(capacity, dtype=DTYPE, device=device)
        self.ep_id = torch.zeros(capacity, dtype=torch.long, device=device)
        self.current_ep = 0
        
        # 🎯 NEW: N-step 버퍼용 로컬 캐시
        self.use_n_step = cfg.use_n_step
        self.n_step = cfg.n_step_size if self.use_n_step else 1
        self.gamma = cfg.gamma
        self.n_step_cache = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        # 덱에 쌓인 정보로 N-step 누적 보상과 진짜 도착 상태(S_next) 계산
        reward, next_state, done = 0.0, self.n_step_cache[-1][3], self.n_step_cache[-1][4]
        for i, transition in enumerate(self.n_step_cache):
            reward += (self.gamma ** i) * transition[2]
            if transition[4]: # 만약 중간에 에피소드가 끝났다면 그 상태로 고정
                next_state, done = transition[3], True
                break
        return reward, next_state, done

    def push(self, s, a, r, s_next, done):
        # 일단 임시 덱에 넣습니다.
        self.n_step_cache.append((s, a, r, s_next, done))
        
        # 덱이 꽉 찼으면 텐서 배열에 기록합니다.
        if len(self.n_step_cache) == self.n_step:
            r_n, s_n, d_n = self._get_n_step_info()
            s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
            self._push_tensor(s_0, a_0, r_n, s_n, d_n)
        
        # 🎯 핵심: 에피소드가 끝났으면(done), 덱에 남아있는 자투리 스텝들도 모두 비워줍니다(Flush).
        if done:
            while len(self.n_step_cache) > 0:
                r_n, s_n, d_n = self._get_n_step_info()
                s_0, a_0 = self.n_step_cache[0][0], self.n_step_cache[0][1]
                self._push_tensor(s_0, a_0, r_n, s_n, d_n)
                self.n_step_cache.popleft()

    def _push_tensor(self, s, a, r, s_next, done):
        idx = self.count % self.capacity
        self.S[idx] = torch.as_tensor(s, dtype=DTYPE, device=self.device)
        self.A[idx] = a; self.R[idx] = r
        self.S_next[idx] = torch.as_tensor(s_next, dtype=DTYPE, device=self.device)
        self.term[idx] = float(done)
        self.ep_id[idx] = self.current_ep
        self.count += 1

    def set_current_episode(self, ep): self.current_ep = ep
    @property
    def current_size(self): return min(self.count, self.capacity)
    @property
    def is_saturated(self): return self.count >= self.capacity
    @property
    def fill_ratio(self): return self.current_size / self.capacity

    def sample_batch(self, batch_size: int) -> Dict:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return {'s': self.S[indices].t(), 'a': self.A[indices], 'r': self.R[indices],
                's_next': self.S_next[indices].t(), 'term': self.term[indices]}

# =========================================================================
# 4. Math Utilities (Batch QR & Triangular Solvers)
# =========================================================================
# =========================================================================
# 4. Math Utilities (Hybrid Batch QR & Triangular Solvers)
# =========================================================================
def tria_operation_batch(A):
    """
    [ND/LD 맞춤형 하이브리드 분해 엔진]
    - Layer Decoupled (LD): 거대 행렬의 병목 해소를 위해 고속 Cholesky 분해 사용
    - Node Decoupled (ND): 작은 행렬의 수치적 안정성을 위해 기존 QR 분해 사용
    """
    # 🚀 [TURBO MODE] Layer Decoupled일 때는 Cholesky 우선 시도
    if cfg.decoupling_mode == 'layer':
        try:
            # 1. 고속 행렬 곱셈: A * A^T 를 통해 양의 정부호 행렬(PD) Y 생성
            Y = torch.bmm(A, A.transpose(-2, -1))
            
            # 2. 수치적 안정성을 위한 미세 Jitter 추가
            jitter = JITTER_TRIA * torch.eye(Y.shape[-1], dtype=A.dtype, device=A.device)
            Y_safe = Y + jitter.unsqueeze(0)
            
            # 3. 고속 숄레스키 분해 (Lower Triangular 반환)
            s = torch.linalg.cholesky(Y_safe)
            return s
            
        except Exception:
            # 만약 특이 행렬(Singular) 문제로 Cholesky가 실패하면,
            # 당황하지 않고 아래의 안전한 QR 로직으로 폴백(Fallback)합니다.
            pass 

    # 🛡️ [SAFE MODE] Node Decoupled 이거나, LD에서 Cholesky가 실패했을 때의 QR 로직
    _, r = torch.linalg.qr(A.transpose(-2, -1).contiguous())
    s = r.transpose(-2, -1).contiguous()  # r은 Upper, s는 Lower Triangular
    
    # 부호 통일 (대각 성분을 양수로 맞춤)
    d = torch.diagonal(s, dim1=-2, dim2=-1)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    s = s * signs.unsqueeze(-2)
    
    # 대각 성분 Clamping (역행렬 계산 시 NaN 폭발 방지)
    d_positive = torch.diagonal(s, dim1=-2, dim2=-1)
    d_clamped = torch.clamp(d_positive, min=JITTER_TRIA)
    s = s - torch.diag_embed(d_positive) + torch.diag_embed(d_clamped)
    
    return s

def safe_inv_tril_batch(L_batch, eye_batch):
    return torch.linalg.solve_triangular(L_batch + JITTER * eye_batch, eye_batch, upper=False)

def robust_solve_spd_batch(S_tril_batch, y_batch, eye_batch):
    S_safe = S_tril_batch + JITTER * eye_batch
    z = torch.linalg.solve_triangular(S_safe, y_batch, upper=False)
    theta = torch.linalg.solve_triangular(S_safe.transpose(-2, -1).contiguous(), z, upper=True)
    return theta

# =========================================================================
# 5. Diagnostic Utilities
# =========================================================================
@torch.no_grad()
def compute_pseudo_cond_from_S(S_batch):
    try:
        S_vals = torch.linalg.svdvals(S_batch)
        S_vals_clamped = S_vals.clamp(min=1e-8)
        Y_eigs = S_vals_clamped ** 2
        y_max_per_neuron = Y_eigs.max(dim=-1).values
        y_min_per_neuron = Y_eigs.min(dim=-1).values
        cond_per_neuron = y_max_per_neuron / y_min_per_neuron.clamp(min=1e-8)
        p_max_per_neuron = 1.0 / y_min_per_neuron.clamp(min=1e-8)
        return (cond_per_neuron.mean().item(), y_max_per_neuron.max().item(), 
                y_min_per_neuron.min().item(), p_max_per_neuron.max().item())
    except Exception:
        return -1.0, -1.0, -1.0, -1.0

@torch.no_grad()
def compute_full_cond_from_S(S_batch):
    try:
        SST = torch.bmm(S_batch, S_batch.transpose(-2, -1))
        eigvals_Y = torch.linalg.eigvalsh(SST)
        y_max = eigvals_Y[:, -1].clamp(min=1e-8)
        y_min = eigvals_Y[:, 0].clamp(min=1e-8)
        cond = y_max / y_min.clamp(min=1e-8)
        return cond.mean().item(), y_max.max().item()
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_effective_rank(X, tol_ratio=1e-3):
    if X.shape[0] > X.shape[1]: X = X.t()
    try:
        X_centered = X - X.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(X_centered)
        s_max = s.max()
        if s_max < 1e-8: return 0.0, 0.0
        eff_rank = (s > s_max * tol_ratio).sum().item()
        stable_rank = (s ** 2).sum().item() / (s_max ** 2).item()
        return float(eff_rank), float(stable_rank)
    except Exception:
        return -1.0, -1.0

@torch.no_grad()
def compute_advantage_null_ratio(theta, info):
    adv_layers = [L for L in info['filter_layers'] if L['type'] in ('advantage', 'q_layer')]
    if not adv_layers: return 0.0, 0.0, 0.0
    a1_layer = adv_layers[-1]
    W_start, W_len = a1_layer['W_start'], a1_layer['W_len']
    b_start, b_len = a1_layer['b_start'], a1_layer['b_len']
    fan_in, fan_out = a1_layer['fan_in'], a1_layer['fan_out']
    
    theta_flat = theta.squeeze()
    W = theta_flat[W_start:W_start + W_len].view(fan_out, fan_in)
    b = theta_flat[b_start:b_start + b_len]
    W_mean = W.mean(dim=0)
    W_dev = W - W_mean.unsqueeze(0)
    
    null_norm = W_mean.norm().item()
    signal_norm = W_dev.norm().item()
    b_mean = b.mean().item()
    b_dev_norm = (b - b.mean()).norm().item()
    
    null_total = (null_norm ** 2 + b_mean ** 2) ** 0.5
    signal_total = (signal_norm ** 2 + b_dev_norm ** 2) ** 0.5
    ratio = null_total / (signal_total + 1e-8)
    return ratio, null_total, signal_total

@torch.no_grad()
def compute_layer_theta_norms(theta, info):
    norms = {}
    theta_flat = theta.squeeze()
    for L, fl in enumerate(info['filter_layers']):
        ltype = fl['type']
        lidx = fl['local_idx']
        label = f"{ltype[0].upper()}{lidx}"
        W_start, W_len = fl['W_start'], fl['W_len']
        b_start, b_len = fl['b_start'], fl['b_len']
        W_norm = theta_flat[W_start:W_start + W_len].norm().item()
        b_norm = theta_flat[b_start:b_start + b_len].norm().item()
        norms[label] = (W_norm ** 2 + b_norm ** 2) ** 0.5
    return norms

@torch.no_grad()
def compute_buffer_diversity(buffer, n_sample=512):
    if buffer.current_size < 32: return None
    n = min(n_sample, buffer.current_size)
    indices = torch.randperm(buffer.current_size, device=buffer.device)[:n]
    states = buffer.S[indices]
    rewards = buffer.R[indices]
    dones = buffer.term[indices]
    ep_ids = buffer.ep_id[indices].float()
    
    state_std = states.std(dim=0).mean().item()
    state_range = (states.max(dim=0).values - states.min(dim=0).values).mean().item()
    done_ratio = dones.mean().item()
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    
    age_min = ep_ids.min().item()
    age_max = ep_ids.max().item()
    age_range_val = age_max - age_min
    age_std = ep_ids.std().item() if n > 1 else 0.0
    
    fill_ratio = buffer.fill_ratio
    is_sat = buffer.is_saturated
    return {
        'state_std': state_std, 'state_range': state_range, 'done_ratio': done_ratio,
        'reward_mean': reward_mean, 'reward_std': reward_std, 'age_min': int(age_min),
        'age_max': int(age_max), 'age_range': age_range_val, 'age_std': age_std,
        'fill_ratio': fill_ratio, 'is_saturated': is_sat,
    }

REF_STATES = torch.tensor([
    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05, 0.0], [0.0, 0.0, -0.05, 0.0],
    [0.0, 0.0, 0.1, 0.5], [0.0, 0.0, -0.1, -0.5],
], dtype=DTYPE)
REF_NAMES = ["balance", "tilt_R", "tilt_L", "fall_R", "fall_L"]

@torch.no_grad()
def compute_ref_q_values(theta, info, normalizer, device):
    ref = REF_STATES.to(device)
    ref_norm = normalizer.normalize(ref) if normalizer else ref
    Q = forward_single(theta.squeeze(), info, ref_norm.t())
    results = {}
    for i, name in enumerate(REF_NAMES):
        q0 = Q[0, i].item()
        q1 = Q[1, i].item()
        dq = q1 - q0
        argmax = 1 if dq > 0 else 0
        results[name] = {'q0': q0, 'q1': q1, 'dq': dq, 'argmax': argmax}
    return results

# =========================================================================
# 6. SRRHUIF Core Functions
# =========================================================================
def _time_update_core(theta_3d, P_sqrt_prev, S_Q_cached, eye_batch, gamma_val):
    combined = torch.cat([P_sqrt_prev, S_Q_cached], dim=2)
    P_sqrt_pred = tria_operation_batch(combined)
    S_pred = safe_inv_tril_batch(P_sqrt_pred, eye_batch)
    
    temp_y = torch.bmm(S_pred.transpose(-2, -1), theta_3d)
    y_pred = torch.bmm(S_pred, temp_y)
    
    scaled_P = gamma_val * P_sqrt_pred
    theta_2d = theta_3d.squeeze(-1)
    X_sigma_all = torch.cat([
        theta_2d.unsqueeze(1),
        theta_2d.unsqueeze(1) + scaled_P.transpose(-2, -1),
        theta_2d.unsqueeze(1) - scaled_P.transpose(-2, -1),
    ], dim=1)
    
    return S_pred, None, y_pred, X_sigma_all, scaled_P

def _compute_ht_core(Z_sigma_T_fwd, Wm_col_fwd, Wc_fwd, zero_col_fwd,
                         scaled_P_fwd, z_measured_exp, S_pred): 
    Z_sigma_T_fwd = Z_sigma_T_fwd.to(DTYPE_FWD)
    Wm_col_fwd = Wm_col_fwd.to(DTYPE_FWD)
    
    z_hat_fwd = torch.bmm(Z_sigma_T_fwd, Wm_col_fwd)
    Z_dev_fwd = Z_sigma_T_fwd - z_hat_fwd
    X_dev_fwd = torch.cat([zero_col_fwd, scaled_P_fwd, -scaled_P_fwd], dim=2)
    P_xz_fwd = torch.bmm(X_dev_fwd * Wc_fwd.view(1, 1, -1), Z_dev_fwd.transpose(1, 2))
    
    z_hat = z_hat_fwd.to(DTYPE)
    residual_all = z_measured_exp.to(DTYPE) - z_hat
    P_xz = P_xz_fwd.to(DTYPE)
    S_pred = S_pred.to(DTYPE)
    
    temp_ht = torch.bmm(S_pred.transpose(-2, -1), P_xz)
    HT_all = torch.bmm(S_pred, temp_ht)
    
    ht_norm = torch.norm(HT_all, dim=1).mean().item()
    resid_norm = torch.norm(residual_all, dim=1).mean().item()
    
    return HT_all, residual_all, z_hat, ht_norm, resid_norm

def _meas_update_core(S_pred, y_pred, HT_all, theta_3d, residual_all, 
                         r_inv_sqrt, r_inv, eye_batch, 
                         tikhonov_lambda=0.1, huber_c=2.0):
    res_abs = torch.abs(residual_all)
    adapt_factor = torch.clamp(res_abs / huber_c, min=1.0)
    
    r_inv_adapt = r_inv / adapt_factor
    r_inv_sqrt_adapt_for_HT = (r_inv_sqrt / torch.sqrt(adapt_factor)).transpose(1, 2)
    tikhonov_sqrt = float(np.sqrt(tikhonov_lambda))
    
    if tikhonov_lambda > 0:
        combined = torch.cat([S_pred, HT_all * r_inv_sqrt_adapt_for_HT, tikhonov_sqrt * eye_batch], dim=2)
    else:
        combined = torch.cat([S_pred, HT_all * r_inv_sqrt_adapt_for_HT], dim=2)

    S_new_all = tria_operation_batch(combined)
    
    ht_theta = torch.bmm(HT_all.transpose(1, 2), theta_3d)
    innov = residual_all + ht_theta
    y_new_all = y_pred + torch.bmm(HT_all, r_inv_adapt * innov)
    
    innov_abs = torch.abs(innov)
    innov_mean, innov_max = torch.mean(innov_abs).item(), torch.max(innov_abs).item()
    
    delta_y = torch.bmm(HT_all, r_inv_adapt * innov)
    delta_y_norm = torch.norm(delta_y, dim=1).mean()
    y_pred_norm = torch.norm(y_pred, dim=1).mean().item()
    y_new_norm = torch.norm(y_new_all, dim=1).mean()

    theta_new_all = robust_solve_spd_batch(S_new_all, y_new_all, eye_batch)
    S_diag = torch.diagonal(S_new_all, dim1=-2, dim2=-1)
    avg_P_new = (1.0 / (S_diag ** 2 + 1e-8)).mean().item()
    
    meas_stats = {
        'innov_mean': innov_mean, 'innov_max': innov_max,
        'resid_in_innov': torch.mean(torch.abs(residual_all)).item(),
        'ht_theta_in_innov': torch.mean(torch.abs(ht_theta)).item(),
        'innov_norm': innov_mean,
        'delta_y': delta_y_norm.item(),
        'y_pred_norm': y_pred_norm,
        'y_new_norm': y_new_norm.item(),
        'avg_P': avg_P_new, 'adapt_ratio': torch.mean(adapt_factor).item()
    }
    return theta_new_all, S_new_all, meas_stats

# =========================================================================
# 7. Initialize theta (Orthogonal / He, config-selectable)
# =========================================================================
def initialize_theta(info, device, cfg):
    """
    Initialize θ vector based on cfg.init_scheme.
    
    'orthogonal': hidden gain=√2, final layer (value/advantage/q_layer 마지막) gain=0.1
                  → Q 분리 잘 됨, UKF 뇌사 방지
    'he':         randn * √(2/fan_in), bias=0
                  → 단순 baseline
    """
    theta = torch.zeros(info['total_params'], dtype=DTYPE, device=device)
    
    for layer in info['layers']:
        fan_in, fan_out = layer['W_shape'][1], layer['W_shape'][0]
        W_len = layer['W_len']
        l_type, l_idx = layer['type'], layer['layer_idx']
        
        if cfg.init_scheme == 'orthogonal':
            W_temp = torch.empty(fan_out, fan_in, dtype=DTYPE, device=device)
            is_final = (
                (l_type == 'value' and l_idx == len(cfg.value_layers)) or
                (l_type == 'advantage' and l_idx == len(cfg.advantage_layers)) or
                (l_type == 'q_layer' and l_idx == len(cfg.q_layers))
            )
            gain = 0.1 if is_final else float(np.sqrt(2.0))
            torch.nn.init.orthogonal_(W_temp, gain=gain)
            theta[layer['W_start']:layer['W_start'] + W_len] = W_temp.view(-1)
        else:  # 'he'
            theta[layer['W_start']:layer['W_start'] + W_len] = \
                torch.randn(W_len, dtype=DTYPE, device=device) * float(np.sqrt(2.0 / fan_in))
        # bias는 0 그대로 (양쪽 공통)
    return theta


def analyze_initial_network(theta, info, env, cfg, normalizer=None, num_samples=100):
    print("\n" + "="*50)
    print(f" 🔍 [초기화 진단] Seed {cfg.seed} 네트워크 해부 리포트")
    print("="*50)
    
    theta_flat = theta.squeeze()
    print(" [1] 레이어별 가중치 분포 (Variance & Scale)")
    for L, fl in enumerate(info['filter_layers']):
        w_start, w_len = fl['W_start'], fl['W_len']
        W = theta_flat[w_start:w_start + w_len]
        
        w_std = W.std().item()
        w_mean = W.mean().item()
        w_max = W.max().item()
        w_min = W.min().item()
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        print(f"  ├─ {label:2s} Layer: Std = {w_std:.4f} | Mean = {w_mean:+.4f} | Range = [{w_min:+.3f}, {w_max:+.3f}]")

    print("\n [2] 초기 Q-Value 신호 대 잡음비 (Dueling Stream)")
    states = []
    for _ in range(num_samples):
        s, _ = env.reset()
        states.append(s)
    states_t = torch.tensor(np.array(states), dtype=DTYPE, device=cfg.device)
    if normalizer: states_t = normalizer.normalize(states_t)
        
    with torch.no_grad():
        Q_initial = forward_single(theta_flat, info, states_t.t()) 
        q0 = Q_initial[0, :]
        q1 = Q_initial[1, :]
        adv_diff = torch.abs(q0 - q1)
        print(f"  ├─ Q(a0) 평균: {q0.mean().item():.4f} (std: {q0.std().item():.4f})")
        print(f"  ├─ Q(a1) 평균: {q1.mean().item():.4f} (std: {q1.std().item():.4f})")
        print(f"  ├─ |Q(a0) - Q(a1)| 평균 차이: {adv_diff.mean().item():.4f} (이 값이 0에 가까우면 UKF 뇌사)")
        print(f"  └─ 초기 행동 쏠림 현상 (a0 선택 비율): {(q0 > q1).float().mean().item() * 100:.1f}%")
    print("="*50 + "\n")

# =========================================================================
# 8. Main SRRHUIF Step (Unified Node/Layer Decoupling)
# =========================================================================
@torch.no_grad()
def srrhuif_step(theta_current_in, theta_target, filter_S_info, batch, sp,
                     is_first, p_init_val, f_cache, frozen_z_measured):
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    
    # ──────────────────────────────────────────────────────────────────
    # [Prior 결정] h=0에서 cfg.h0_prior_source에 따라 분기
    #   'target' = target net (기존 동작)
    #   'init'   = 학습 시작시 frozen된 θ_init (FIR 정신)
    # h≥1: 항상 직전 추정치 (theta_current_in) 사용
    # ──────────────────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_prior = sp['theta_init'].clone()
        else:  # 'target'
            theta_prior = theta_target.clone()
    else:
        theta_prior = theta_current_in.clone()
    
    theta_current = theta_current_in.clone()
    new_S_info_dict = {}
    total_loss, layer_count = 0.0, 0

    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)

    unified = f_cache.unified_thetas
    # ──────────────────────────────────────────────────────────────────
    # [핵심 버그 fix] unified base를 theta_prior로
    #   기존 버그: theta_current로 깔아 h=0에서 *다른 레이어*가 t-1 추정값을
    #   prior로 쓰는 inconsistency 발생.
    #   Fix: theta_prior로 깔면 h=0에서 모든 레이어 base 일관 (target or init).
    #        h≥1에선 theta_prior == theta_current_in이라 동작 동일 (no-op).
    # ──────────────────────────────────────────────────────────────────
    unified[:] = theta_prior.squeeze().to(DTYPE_FWD)

    per_layer = {}
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        lc = f_cache.get(L)
        
        # [핵심 매핑] Mode에 따라 Prior를 추출
        if cfg.decoupling_mode == 'node':
            W_prior = theta_prior.squeeze()[fl['W_start']:fl['W_start'] + fl['W_len']].view(fl['fan_out'], fl['fan_in'])
            b_prior = theta_prior.squeeze()[fl['b_start']:fl['b_start'] + fl['b_len']]
            theta_all_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1) # [fan_out, fan_in+1]
        else:
            theta_all_prior = theta_prior.squeeze()[fl['W_start']:fl['W_start'] + fl['param_len']].unsqueeze(0) # [1, param_len]
            
        theta_all_prior_3d = theta_all_prior.unsqueeze(-1)

        S_3d = filter_S_info[L]
        if is_first or S_3d is None:
            P_sqrt_prev = np.sqrt(p_init_val) * lc['eye_block_batch'].clone()
        else:
            P_sqrt_prev = safe_inv_tril_batch(S_3d.permute(2, 0, 1), lc['eye_block_batch'])

        per_layer[L] = {
            'fl': fl, 'lc': lc, 'theta_all_prior': theta_all_prior,
            'theta_all_prior_3d': theta_all_prior_3d, 'P_sqrt_prev': P_sqrt_prev,
        }
        
    current_q_std = sp.get('current_q_std', cfg.q_init)
    current_r_std = sp.get('current_r_std', cfg.r_init)
    current_r_inv_sqrt = 1.0 / current_r_std
    current_r_inv = 1.0 / (current_r_std ** 2)
    
    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_P_sqrt = torch.cat([per_layer[L]['P_sqrt_prev'] for L in layers_in_grp], dim=0)
        dynamic_S_Q = current_q_std * grp['eye_grouped']

        S_pred_g, _, y_pred_g, X_sigma_g, scaled_P_g = _time_update_core(
            all_theta_3d, all_P_sqrt, dynamic_S_Q, grp['eye_grouped'], grp['gamma'])
        
        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            per_layer[L]['S_pred'] = S_pred_g[s:e]
            per_layer[L]['y_pred'] = y_pred_g[s:e]
            per_layer[L]['X_sigma_all'] = X_sigma_g[s:e]
            per_layer[L]['scaled_P'] = scaled_P_g[s:e]

    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        X_sigma_f32 = pl['X_sigma_all'].to(DTYPE_FWD)
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        # [핵심 매핑] Mode에 따라 Sigma Point 흩뿌리기
        if cfg.decoupling_mode == 'node':
            layer_view = unified[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], -1)
            layer_view.scatter_(dim=2, index=lc['w_col_idx'], src=X_sigma_f32[:, :, :fl['fan_in']])
            layer_view.scatter_(dim=2, index=lc['b_col_idx'], src=X_sigma_f32[:, :, fl['fan_in']:fl['fan_in'] + 1])
        else:
            # [버그 수정됨] X_sigma_f32[0] 을 그대로 복사 ([num_sigma, param_len])
            unified[fwd_start:fwd_end, fl['W_start']:fl['W_start'] + fl['param_len']] = X_sigma_f32[0]

    
    z_measured = frozen_z_measured
    target_var = torch.var(z_measured).item()

    Q_all_f32 = forward_bmm(unified, info, s_batch)
    
    for L in range(info['num_filter_layers']):
        pl = per_layer[L]
        lc, fl = pl['lc'], pl['fl']
        fwd_start, fwd_end = f_cache.layer_fwd_slices[L]
        
        Q_L_f32 = Q_all_f32[fwd_start:fwd_end].view(lc['num_blocks'], lc['num_sigma'], info['nA'], -1)
        Z_sigma_T_f32 = Q_L_f32[:, :, batch['a'], torch.arange(batch_sz, device=device)].transpose(1, 2)
        z_measured_exp = z_measured.unsqueeze(0).expand(lc['num_blocks'], -1, -1)

        HT_all, residual_all, z_hat, ht_norm, resid_norm = _compute_ht_core(
            Z_sigma_T_f32, lc['Wm_col_f32'], lc['Wc_f32'], lc['zero_col_f32'],
            pl['scaled_P'].to(DTYPE_FWD), z_measured_exp, pl['S_pred'])

        per_layer[L]['HT_all'] = HT_all
        per_layer[L]['residual_all'] = residual_all
        per_layer[L]['loss'] = torch.mean(residual_all ** 2)
        per_layer[L]['ht_norm'] = ht_norm
        per_layer[L]['resid_norm'] = resid_norm
        per_layer[L]['resid_max'] = torch.max(torch.abs(residual_all)).item()
        layer_count += 1

    total_innov_mean, total_innov_max = 0.0, 0.0
    total_ht_norm, total_resid_norm = 0.0, 0.0
    total_delta_y, total_y_new, total_avg_P = 0.0, 0.0, 0.0
    total_resid_in_innov, total_ht_theta_in_innov = 0.0, 0.0
    total_innov_norm, total_y_pred_norm, total_adapt_ratio = 0.0, 0.0, 0.0
    group_count = 0
    
    per_layer_cond, per_layer_ymax, per_layer_cond_full = {}, {}, {}

    for bs_val, grp in f_cache.block_groups.items():
        layers_in_grp = grp['layers']
        offsets = grp['offsets']

        all_S_pred = torch.cat([per_layer[L]['S_pred'] for L in layers_in_grp], dim=0)
        all_y_pred = torch.cat([per_layer[L]['y_pred'] for L in layers_in_grp], dim=0)
        all_HT = torch.cat([per_layer[L]['HT_all'] for L in layers_in_grp], dim=0)
        all_theta_3d = torch.cat([per_layer[L]['theta_all_prior_3d'] for L in layers_in_grp], dim=0)
        all_residual = torch.cat([per_layer[L]['residual_all'] for L in layers_in_grp], dim=0)

        # [Trust region 제거] 항상 표준 measurement update만 사용
        theta_new_g, S_new_g, meas_stats = _meas_update_core(
            all_S_pred, all_y_pred, all_HT, all_theta_3d,
            all_residual, current_r_inv_sqrt, current_r_inv, grp['eye_grouped'],
            tikhonov_lambda=cfg.tikhonov_lambda, huber_c=cfg.huber_c)

        total_innov_mean += meas_stats['innov_mean']
        total_innov_max = max(total_innov_max, meas_stats['innov_max'])
        total_delta_y += meas_stats['delta_y']
        total_y_new += meas_stats['y_new_norm']
        total_avg_P += meas_stats['avg_P']
        total_resid_in_innov += meas_stats['resid_in_innov']
        total_ht_theta_in_innov += meas_stats['ht_theta_in_innov']
        total_innov_norm += meas_stats['innov_norm']
        total_y_pred_norm += meas_stats['y_pred_norm']
        total_adapt_ratio += meas_stats['adapt_ratio']
        group_count += 1
            
        for L in layers_in_grp:
            total_ht_norm += per_layer[L]['ht_norm']
            total_resid_norm += per_layer[L]['resid_norm']

        for i, L in enumerate(layers_in_grp):
            s, e = offsets[i], offsets[i + 1]
            pl = per_layer[L]
            fl = pl['fl']
            theta_new_L = theta_new_g[s:e]
            S_new_L = S_new_g[s:e]
            
            if cfg.diag_horizon_cond:
                label = f"{fl['type'][0].upper()}{fl['local_idx']}"
                cond_val, ymax_val, _, _ = compute_pseudo_cond_from_S(S_new_L)
                per_layer_cond[label] = cond_val
                per_layer_ymax[label] = ymax_val
                if cfg.use_full_eigvalsh:
                    full_cond, _ = compute_full_cond_from_S(S_new_L)
                    per_layer_cond_full[label] = full_cond

            invalid = ~torch.isfinite(theta_new_L).all(dim=(1, 2))
            if invalid.any(): theta_new_L[invalid] = pl['theta_all_prior'][invalid].unsqueeze(-1)

            theta_flat = theta_current.squeeze()
            
            if cfg.decoupling_mode == 'node':
                W_new = theta_new_L[:, :fl['fan_in'], 0]
                b_new = theta_new_L[:, fl['fan_in'], 0]
                
                if cfg.max_layer_step > 0:
                    W_curr = theta_flat[fl['W_start']:fl['W_start']+fl['W_len']].view(fl['fan_out'], fl['fan_in'])
                    b_curr = theta_flat[fl['b_start']:fl['b_start']+fl['b_len']]
                    delta_norm = torch.sqrt(torch.norm(W_new - W_curr)**2 + torch.norm(b_new - b_curr)**2)
                    if delta_norm > cfg.max_layer_step:
                        scale = cfg.max_layer_step / (delta_norm + 1e-8)
                        W_new = W_curr + (W_new - W_curr) * scale
                        b_new = b_curr + (b_new - b_curr) * scale
                        
                theta_flat[fl['W_start']:fl['W_start'] + fl['W_len']] = W_new.reshape(-1)
                theta_flat[fl['b_start']:fl['b_start'] + fl['b_len']] = b_new
            else:
                theta_new_flat = theta_new_L[0, :, 0]
                if cfg.max_layer_step > 0:
                    theta_curr = theta_flat[fl['W_start']:fl['W_start'] + fl['param_len']]
                    delta_norm = torch.norm(theta_new_flat - theta_curr)
                    if delta_norm > cfg.max_layer_step:
                        scale = cfg.max_layer_step / (delta_norm + 1e-8)
                        theta_new_flat = theta_curr + (theta_new_flat - theta_curr) * scale
                theta_flat[fl['W_start']:fl['W_start'] + fl['param_len']] = theta_new_flat
                
            theta_current = theta_flat.view(-1, 1)
            new_S_info_dict[L] = S_new_L.permute(1, 2, 0)
            total_loss += pl['loss']
            
    new_S_info = [new_S_info_dict[L] for L in range(info['num_filter_layers'])]
    delta_theta = theta_current.squeeze() - theta_current_in.squeeze()
    k_gain_norm = torch.norm(delta_theta).item()
    
    if cfg.max_k_gain > 0 and k_gain_norm > cfg.max_k_gain:
        scale = cfg.max_k_gain / k_gain_norm
        theta_current = (theta_current_in.squeeze() + delta_theta * scale).view(-1, 1)
        k_gain_norm = cfg.max_k_gain

    per_layer_ht, per_layer_delta, per_layer_resid_max = {}, {}, {}
    theta_new_flat = theta_current.squeeze()
    theta_old_flat = theta_current_in.squeeze()
    for L in range(info['num_filter_layers']):
        fl = info['filter_layers'][L]
        label = f"{fl['type'][0].upper()}{fl['local_idx']}"
        per_layer_ht[label] = per_layer[L]['ht_norm']
        per_layer_resid_max[label] = per_layer[L]['resid_max']
        s, p_len = fl['W_start'], fl['param_len']
        per_layer_delta[label] = torch.norm(theta_new_flat[s:s+p_len] - theta_old_flat[s:s+p_len]).item()
        
    n_layers = info['num_filter_layers']
    gc = max(group_count, 1)
    dbg = {
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
        'per_layer_resid_max': per_layer_resid_max,
        'per_layer_cond': per_layer_cond,
        'per_layer_ymax': per_layer_ymax,
        'per_layer_cond_full': per_layer_cond_full,
        'adapt_ratio': total_adapt_ratio / gc,
    }
    
    return theta_current, new_S_info, (total_loss / layer_count).item(), target_var, k_gain_norm, dbg


# =========================================================================
# 8b. SRRHUIF Full Vector Mode
# =========================================================================
def srrhuif_step_fv(theta_current_in, theta_target, filter_S_info, batch, sp,
                    is_first, p_init_val, fv_cache, frozen_z_measured):
    """
    Full Vector mode: 전체 θ ∈ R^n_x를 하나의 블록으로 다룸.
    - layer/node 분리 없음 (모든 파라미터 covariance가 한 행렬에 들어감)
    - 비용: O(n_x²) 메모리, O(n_x³) QR. CartPole 토이 스케일에서만 권장.
    - 진동 측면에선 가장 정확 (within-layer + between-layer correlation 모두 잡힘)
    """
    device, info, batch_sz = sp['device'], sp['info'], sp['batch_sz']
    n_x = info['total_params']
    
    # ─────────────────────────────────────────────────────────────
    # [Prior 결정] h=0이면 cfg.h0_prior_source에 따라 분기
    # ─────────────────────────────────────────────────────────────
    if is_first:
        if cfg.h0_prior_source == 'init':
            theta_pred = sp['theta_init'].clone()
        else:  # 'target'
            theta_pred = theta_target.clone()
    else:
        theta_pred = theta_current_in.clone()
    
    theta_pred_flat = theta_pred.squeeze()  # [n_x]
    
    s_batch, s_next = batch['s'].t(), batch['s_next'].t()
    if sp.get('normalizer'):
        s_batch = sp['normalizer'].normalize(s_batch)
        s_next = sp['normalizer'].normalize(s_next)
    
    # ─────────────────────────────────────────────────────────────
    # [A] Time Update (prediction)
    # ─────────────────────────────────────────────────────────────
    eye_n = fv_cache.eye_n
    if is_first or filter_S_info is None:
        P_sqrt_prev = float(np.sqrt(p_init_val)) * eye_n
    else:
        # filter_S_info: [n_x, n_x] lower-triangular S (information factor)
        P_sqrt_prev = safe_inv_tril_batch(
            filter_S_info.unsqueeze(0), eye_n.unsqueeze(0)
        ).squeeze(0)
    
    S_Q = cfg.q_init * eye_n
    P_sqrt_pred = tria_operation_batch(
        torch.cat([P_sqrt_prev, S_Q], dim=1).unsqueeze(0)
    ).squeeze(0)  # [n_x, n_x]
    
    S_pred = safe_inv_tril_batch(
        P_sqrt_pred.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)  # [n_x, n_x] = P^{-1/2}_{t|t-1}
    Y_pred = S_pred @ S_pred.t()
    y_pred = Y_pred @ theta_pred_flat.unsqueeze(-1)  # [n_x, 1]
    
    # ─────────────────────────────────────────────────────────────
    # [B] Sigma Point Generation: 2n_x+1 points around theta_pred
    # ─────────────────────────────────────────────────────────────
    scaled_P = fv_cache.gamma_sigma * P_sqrt_pred  # [n_x, n_x]
    unified = fv_cache.unified_thetas  # [num_sigma, n_x]
    unified[0] = theta_pred_flat.to(DTYPE_FWD)
    unified[1:n_x+1] = (theta_pred_flat.unsqueeze(0) + scaled_P.t()).to(DTYPE_FWD)
    unified[n_x+1:] = (theta_pred_flat.unsqueeze(0) - scaled_P.t()).to(DTYPE_FWD)
    
    # ─────────────────────────────────────────────────────────────
    # [C] Forward all sigma points → measurement statistics
    # ─────────────────────────────────────────────────────────────
    Q_all_f32 = forward_bmm(unified, info, s_batch)  # [num_sigma, nA, batch_sz]
    Z_sigma_T_f32 = Q_all_f32[:, batch['a'], torch.arange(batch_sz, device=device)]  # [num_sigma, batch_sz]
    Z_sigma_T = Z_sigma_T_f32.to(DTYPE)
    
    # z_hat = Σ Wm·Z_sigma  (predicted measurement mean)
    z_hat = (fv_cache.Wm.view(-1, 1) * Z_sigma_T).sum(dim=0, keepdim=True).t()  # [batch_sz, 1]
    
    # ─────────────────────────────────────────────────────────────
    # [D] DDQN target value (using cached target Q for next state)
    # ─────────────────────────────────────────────────────────────
    
    z_measured = frozen_z_measured
    target_var = torch.var(z_measured).item()
    
    # ─────────────────────────────────────────────────────────────
    # [E] Statistical linearization → H^T
    # ─────────────────────────────────────────────────────────────
    # X_dev: deviation of sigma points from center. [num_sigma, n_x]
    X_dev = torch.zeros(fv_cache.num_sigma, n_x, dtype=DTYPE, device=device)
    X_dev[1:n_x+1] = scaled_P.t()
    X_dev[n_x+1:] = -scaled_P.t()
    
    Z_dev = Z_sigma_T - z_hat.t()  # [num_sigma, batch_sz]
    P_xz = (X_dev * fv_cache.Wc.view(-1, 1)).t() @ Z_dev  # [n_x, batch_sz]
    HT = Y_pred @ P_xz  # [n_x, batch_sz]
    
    residual = z_measured - z_hat  # [batch_sz, 1]
    loss = torch.mean(residual ** 2)
    
    # ─────────────────────────────────────────────────────────────
    # [F] Information form measurement update
    # ─────────────────────────────────────────────────────────────
    # Adaptive R (스케줄링) — sp['current_r_std']가 있으면 그걸, 없으면 cfg.r_init
    current_r_std = sp.get('current_r_std', cfg.r_init)
    r_inv = 1.0 / (current_r_std ** 2)
    r_inv_sqrt = 1.0 / current_r_std
    
    # Huber-style adaptive R
    res_abs = torch.abs(residual)
    adapt_factor = torch.clamp(res_abs / cfg.huber_c, min=1.0)  # [batch_sz, 1]
    r_inv_adapt = r_inv / adapt_factor  # [batch_sz, 1]
    r_inv_sqrt_adapt = (r_inv_sqrt / torch.sqrt(adapt_factor)).t()  # [1, batch_sz]
    
    # S_new = QR([S_pred | HT * r_inv_sqrt_adapt | sqrt(λ)·I])
    tikhonov_sqrt = float(np.sqrt(cfg.tikhonov_lambda))
    if cfg.tikhonov_lambda > 0:
        combined = torch.cat([S_pred, HT * r_inv_sqrt_adapt, tikhonov_sqrt * eye_n], dim=1)
    else:
        combined = torch.cat([S_pred, HT * r_inv_sqrt_adapt], dim=1)
    S_new = tria_operation_batch(combined.unsqueeze(0)).squeeze(0)
    
    # y_new = y_pred + HT · r_inv_adapt · (residual + HT^T · θ_pred)
    ht_theta = HT.t() @ theta_pred_flat.unsqueeze(-1)  # [batch_sz, 1]
    innov = residual + ht_theta
    y_new = y_pred + HT @ (r_inv_adapt * innov)
    
    # Recover θ from information form
    theta_new = robust_solve_spd_batch(
        S_new.unsqueeze(0), y_new.unsqueeze(0), eye_n.unsqueeze(0)
    ).squeeze(0)  # [n_x, 1]
    
    if not torch.isfinite(theta_new).all():
        theta_new = theta_pred.clone()
    
    # ─────────────────────────────────────────────────────────────
    # [G] Diagnostics (LD/ND와 호환되는 dbg dict)
    # ─────────────────────────────────────────────────────────────
    # K-gain norm: ||HT · r_inv||
    k_gain = HT * r_inv_sqrt_adapt
    k_gain_norm = torch.norm(k_gain).item()
    
    # avg_P: 1/diag(Y_new)의 평균 (대략 평균 분산)
    Y_new = S_new @ S_new.t()
    Y_diag = torch.diagonal(Y_new)
    avg_P = (1.0 / (Y_diag + 1e-8)).mean().item()
    
    target_var = torch.var(z_measured).item()
    
    # innov 분해 stats
    innov_abs = torch.abs(innov)
    resid_abs = torch.abs(residual)
    ht_theta_abs = torch.abs(ht_theta)
    
    delta_theta_norm = torch.norm(theta_new - theta_pred).item()
    
    dbg = {
        'innov_mean': innov_abs.mean().item(),
        'innov_max': innov_abs.max().item(),
        'innov_norm': innov_abs.mean().item(),
        'resid_in_innov': resid_abs.mean().item(),
        'ht_theta_in_innov': ht_theta_abs.mean().item(),
        'avg_P': avg_P,
        'ht_norm': torch.norm(HT).item(),
        'resid_norm': torch.norm(residual).item(),
        'delta_y': torch.norm(HT @ (r_inv_adapt * innov)).item(),
        'y_pred_norm': torch.norm(y_pred).item(),
        'y_new_norm': torch.norm(y_new).item(),
        'adapt_ratio': adapt_factor.mean().item(),
        # FV는 layer 분리 없으니 per_layer dict는 비움 (training loop이 .get() 패턴 사용)
        'per_layer_ht': {'fv': torch.norm(HT).item()},
        'per_layer_delta': {'fv': delta_theta_norm},
        'per_layer_resid_max': {'fv': resid_abs.max().item()},
        'per_layer_cond': {'fv': 1.0},        # placeholder
        'per_layer_ymax': {'fv': torch.max(torch.abs(y_new)).item()},
        'per_layer_cond_full': {'fv': 1.0},
    }
    
    return theta_new, S_new, loss.item(), target_var, k_gain_norm, dbg


# =========================================================================
# 9. Live Plotter
# =========================================================================
class LivePlotter:
    def __init__(self, method_name: str, max_episodes: int, param_str: str = ""):
        self.method_name = method_name
        self.outdir = cfg.outdir
        
        self.rewards, self.losses, self.p_inits, self.z_vars = [], [], [], []
        self.k_gains = [] 
        self.q_vals_0, self.q_vals_1 = [], []
        self.total_time, self.avg_step_time = 0.0, 0.0
        
        self.cond_history = {}
        self.ymax_history = {}
        self.theta_norm_history = {}
        self.null_ratio_history = []
        self.eff_rank_history = []
        self.stable_rank_history = []
        self.argmax_flip_history = []
        self.ref_dq_history = {name: [] for name in REF_NAMES}
        
        self.buf_state_std, self.buf_state_range, self.buf_done_ratio = [], [], []
        self.buf_reward_std, self.buf_fill_ratio, self.buf_age_range, self.buf_age_std = [], [], [], []
        self.buf_saturated_ep = None  
        
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
        self.ax_p.set_title('P_init (Fixed)'); self.ax_p.set_xlim(0, max_episodes)
        self.ax_p.set_ylim(0, cfg.p_init * 1.5)
            
        self.ax_z = self.axes[3]
        self.line_z, = self.ax_z.plot([], [], 'm-', linewidth=1.5)
        self.ax_z.set_title('TD Target Variance (Z_var)'); self.ax_z.set_xlim(0, max_episodes)

        self.ax_k = self.axes[4]
        self.line_k, = self.ax_k.plot([], [], 'darkorange', linewidth=1.5)
        self.ax_k.set_title('Weight Update Norm ||Δθ||'); self.ax_k.set_xlim(0, max_episodes)

        self.ax_q = self.axes[5]
        self.line_q0, = self.ax_q.plot([], [], 'c-', linewidth=1.5, label='Q(a=0) Left')
        self.line_q1, = self.ax_q.plot([], [], 'm-', linewidth=1.5, label='Q(a=1) Right')
        self.ax_q.set_title('Avg Q-Values'); self.ax_q.set_xlim(0, max_episodes)
        self.ax_q.legend(loc='upper left')
        
        plt.tight_layout()
        prefix = f"{param_str}_" if param_str else ""
        clean_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        self.filename = os.path.join(self.outdir, f"{prefix}{clean_name}")
    
    def add(self, reward, loss, p_init=0.0, z_var=0.0, k_gain=0.0, q0=0.0, q1=0.0): 
        self.rewards.append(reward)
        self.losses.append(max(loss, 1e-8))
        self.p_inits.append(p_init)
        self.z_vars.append(z_var)
        self.k_gains.append(k_gain) 
        self.q_vals_0.append(q0)
        self.q_vals_1.append(q1)
    
    def add_diagnostics(self, cond_dict, ymax_dict, theta_norms, null_ratio,
                        eff_rank, stable_rank, argmax_flip, ref_q):
        if cond_dict:
            for k, v in cond_dict.items(): self.cond_history.setdefault(k, []).append(v)
        if ymax_dict:
            for k, v in ymax_dict.items(): self.ymax_history.setdefault(k, []).append(v)
        if theta_norms:
            for k, v in theta_norms.items(): self.theta_norm_history.setdefault(k, []).append(v)
        self.null_ratio_history.append(null_ratio)
        self.eff_rank_history.append(eff_rank)
        self.stable_rank_history.append(stable_rank)
        self.argmax_flip_history.append(argmax_flip)
        if ref_q:
            for name in REF_NAMES: self.ref_dq_history[name].append(ref_q[name]['dq'])
    
    def add_buffer_diag(self, buf_info, ep):
        if buf_info is None:
            self.buf_state_std.append(float('nan')); self.buf_state_range.append(float('nan'))
            self.buf_done_ratio.append(float('nan')); self.buf_reward_std.append(float('nan'))
            self.buf_fill_ratio.append(0.0); self.buf_age_range.append(0); self.buf_age_std.append(0.0)
            return
        self.buf_state_std.append(buf_info['state_std']); self.buf_state_range.append(buf_info['state_range'])
        self.buf_done_ratio.append(buf_info['done_ratio']); self.buf_reward_std.append(buf_info['reward_std'])
        self.buf_fill_ratio.append(buf_info['fill_ratio']); self.buf_age_range.append(buf_info['age_range'])
        self.buf_age_std.append(buf_info['age_std'])
        if buf_info['is_saturated'] and self.buf_saturated_ep is None: self.buf_saturated_ep = ep
    
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
        
        for ax in self.axes: ax.relim(); ax.autoscale_view()
        self.axes[0].set_ylim(0, max(max(self.rewards, default=520) * 1.1, 520))
        plt.savefig(f'{self.filename}_live.png', dpi=100)
    
    def save_diagnostic_plots(self):
        if not self.cond_history and not self.theta_norm_history: return
        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
        ax = axes[0, 0]
        for label, vals in sorted(self.cond_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log'); ax.set_title('Pseudo Condition Number per Layer')
        ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        for label, vals in sorted(self.ymax_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_yscale('log'); ax.set_title('Y_max per Layer (max eigenvalue approx)')
        ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        for label, vals in sorted(self.theta_norm_history.items()): ax.plot(vals, label=label, linewidth=1.5)
        ax.set_title('Layer ||θ|| Evolution'); ax.set_xlabel('Episode'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(self.null_ratio_history, 'r-', linewidth=2)
        ax.set_title('Advantage/Q-Layer Null/Signal Ratio'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(self.eff_rank_history, 'b-', linewidth=2, label='effective rank')
        ax.plot(self.stable_rank_history, 'g--', linewidth=2, label='stable rank')
        ax.set_title('Shared Output Rank'); ax.set_xlabel('Episode'); ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        for name in REF_NAMES: ax.plot(self.ref_dq_history[name], label=name, linewidth=1.5)
        ax.set_title('ΔQ = Q(right) - Q(left) at Reference States'); ax.set_xlabel('Episode')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3); ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.savefig(f'{self.filename}_diagnostics.png', dpi=120, bbox_inches='tight'); plt.close(fig)
        
        if self.argmax_flip_history:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(self.argmax_flip_history, 'orange', linewidth=1.5)
            ax2.set_title('Argmax Flip Rate (update-induced policy instability)'); ax2.set_xlabel('Episode')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
            ax2.grid(True, alpha=0.3); ax2.legend(); plt.tight_layout()
            plt.savefig(f'{self.filename}_argmax_flip.png', dpi=120); plt.close(fig2)
        
        if self.buf_state_std and any(not (np.isnan(v)) for v in self.buf_state_std):
            fig3, axes3 = plt.subplots(2, 3, figsize=(21, 10))
            ax = axes3[0, 0]
            ax.plot(self.buf_fill_ratio, 'b-', linewidth=2)
            if self.buf_saturated_ep is not None:
                ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', label=f'First saturation: Ep {self.buf_saturated_ep}')
                ax.legend()
            ax.set_title('Buffer Fill Ratio'); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
            
            ax = axes3[0, 1]
            ax.plot(self.buf_state_std, 'g-', linewidth=2, label='state_std')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('State Std (diversity of sampled states)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes3[0, 2]
            ax.plot(self.buf_state_range, 'm-', linewidth=2, label='state_range')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('State Range (max-min)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 0]
            ax.plot(self.buf_done_ratio, 'orange', linewidth=2)
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Done Ratio in Buffer'); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 1]
            ax.plot(self.buf_reward_std, 'purple', linewidth=2)
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Reward Std in Buffer'); ax.grid(True, alpha=0.3)
            
            ax = axes3[1, 2]
            ax.plot(self.buf_age_range, 'teal', linewidth=2, label='age range')
            ax.plot(self.buf_age_std, 'brown', linewidth=2, label='age std')
            if self.buf_saturated_ep is not None: ax.axvline(x=self.buf_saturated_ep, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Buffer Age Diversity (ep_id range/std)'); ax.legend(); ax.grid(True, alpha=0.3)
            
            plt.tight_layout(); plt.savefig(f'{self.filename}_buffer_diag.png', dpi=120, bbox_inches='tight'); plt.close(fig3)
    
    def close(self):
        plt.close(self.fig)

# =========================================================================
# 10. Landscape Visualization
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


# =========================================================================
# 11. Main Loop with Full Logging
# =========================================================================
def train_srrhuif():
    net_seed = cfg.network_seed if cfg.network_seed is not None else cfg.seed
    env_seed = cfg.env_seed if cfg.env_seed is not None else cfg.seed
    set_all_seeds(net_seed)
    env = gym.make(cfg.env_name)
    env.action_space.seed(net_seed)
    dimS, nA = env.observation_space.shape[0], env.action_space.n
    info = create_network_info(dimS, nA, cfg)
    
    method_title = f"{'D3QN' if cfg.use_dueling else 'DDQN'} + {cfg.decoupling_mode.upper()} Decoupled"
    
    print(f"\n{'='*60}")
    print(f"  SRRHUIF-ND/LD v12.0 ({method_title}) Robust Session")
    print(f"  Horizon: {cfg.N_horizon} | Batch: {cfg.batch_size} | Params: {info['total_params']}")
    print(f"  Settings: P_init={cfg.p_init}, Tikhonov={cfg.tikhonov_lambda}, Huber_c={cfg.huber_c}")
    print(f"  Output Dir: {cfg.outdir}")
    print(f"  Seeds: network={net_seed}, env={env_seed}")
    print(f"{'='*60}\n")

    normalizer = InputNormalizer(cfg.device) if cfg.use_input_norm else None
    
    # ──────────────────────────────────────────────────────────────────
    # [FV 분기] 모드별 캐시 생성
    # ──────────────────────────────────────────────────────────────────
    is_fv = (cfg.decoupling_mode == 'fv')
    if is_fv:
        f_cache = FilterCacheFV(info, cfg, cfg.device)
    else:
        f_cache = FilterCache(info, cfg, cfg.device)
    
    sp = {'info': info, 'n_x': info['total_params'], 'batch_sz': cfg.batch_size, 'normalizer': normalizer, 'device': cfg.device}

    theta = initialize_theta(info, cfg.device, cfg).view(-1, 1)
    # ──────────────────────────────────────────────────────────────────
    # [FIR 철학] theta_init: 학습 시작시 한 번 뽑은 frozen 값.
    #   cfg.h0_prior_source == 'init'이면 매 horizon의 h=0에서 prior로 사용.
    # ──────────────────────────────────────────────────────────────────
    theta_init = theta.clone()
    sp['theta_init'] = theta_init
    
    theta_target = theta.clone()
    
    print(f"[Init] scheme='{cfg.init_scheme}' | h0_prior_source='{cfg.h0_prior_source}' | mode='{cfg.decoupling_mode}'")
    analyze_initial_network(theta, info, env, cfg, normalizer)
    
    # FV 모드: filter_S_info는 단일 텐서 (None으로 시작)
    # node/layer 모드: 레이어별 dict
    if is_fv:
        filter_S_info = None
    else:
        filter_S_info = [None] * info['num_filter_layers']
    buffer = TensorReplayBuffer(cfg.buffer_size, dimS, cfg.device, cfg) 
    s_t_buffer = torch.empty(dimS, dtype=DTYPE, device=cfg.device)

    batch_hist = deque(maxlen=cfg.N_horizon)
    
    logger = LivePlotter(method_title, cfg.max_episodes, cfg.param_str)
    
    steps_done = 0
    train_start_time = time.time()
    update_times = []
    
    prev_ep_delta = None
    prev_buf_saturated = False
    theta_ep_start = theta.squeeze().clone()

    s, _ = env.reset(seed=env_seed)
    
    for ep in range(1, cfg.max_episodes + 1):
        s, _ = env.reset(seed=env_seed + ep)
        buffer.set_current_episode(ep)
        
        ep_r, ep_l, ep_var, ep_k_gain, ep_start = 0, [], [], [], time.time()
        ep_q0, ep_q1 = [], []
        ep_i_mean, ep_i_max = [], []
        
        last_h_k_traj, last_h_p_traj, last_h_ht_traj = [], [], []
        last_h_resid_traj, last_h_innov_decomp, last_h_cos_traj = [], [], []
        last_h_layer_ht, last_h_layer_delta = [], []
        last_h_layer_resid_max = [] 
        last_h_layer_cond, last_h_layer_ymax = [], []
        last_ep_cos = None
        
        ep_cond_collect, ep_ymax_collect, ep_argmax_flips = {}, {}, []
        theta_ep_start = theta.squeeze().clone()
        
        for t in range(cfg.max_steps):
            steps_done += 1
            if steps_done <= cfg.warmup_step:
                eps = 1.0 
            else:
                active_steps = steps_done - cfg.warmup_step
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-active_steps / cfg.eps_decay_steps)
            
            with torch.no_grad():
                s_t_buffer.copy_(torch.as_tensor(s, dtype=DTYPE))
                s_t = s_t_buffer
                if normalizer: s_t = normalizer.normalize(s_t)
                q_vals = forward_single(theta.squeeze(), info, s_t).squeeze()
                ep_q0.append(q_vals[0].item())
                ep_q1.append(q_vals[1].item())

            a = env.action_space.sample() if np.random.rand() < eps else q_vals.argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            buffer.push(s, a, r / cfg.scale_factor, ns, done)
            s, ep_r = ns, ep_r + r

            if steps_done > cfg.warmup_step and buffer.current_size >= cfg.batch_size and steps_done % cfg.update_interval == 0:
                update_start = time.perf_counter()
                batch = buffer.sample_batch(cfg.batch_size)
                batch_hist.append(batch)
                
                if len(batch_hist) == cfg.N_horizon:
                    h_k_traj, h_p_traj, h_ht_traj = [], [], []
                    h_resid_traj, h_resid_in_innov_traj, h_ht_theta_traj = [], [], []
                    h_innov_traj, h_cos_traj = [], []
                    h_layer_ht, h_layer_delta, h_layer_cond, h_layer_ymax = [], [], [], []
                    h_layer_resid_max = [] 
                    prev_h_delta = None
                    q_next_caches = []
                    
                    frozen_targets = []
                    with torch.no_grad():
                        for h in range(cfg.N_horizon):
                            batch_h = batch_hist[h]
                            s_next_h = batch_h['s_next'].t()
                            if normalizer: 
                                s_next_h = normalizer.normalize(s_next_h)
                            
                            # 1. 온라인 뇌(theta)로 행동 평가 (Standard DDQN)
                            Q_online = forward_single(theta.squeeze(), info, s_next_h)
                            a_best_next = Q_online.argmax(dim=0)
                            
                            # 2. 타겟 뇌(theta_target)로 가치 평가
                            Q_tgt = forward_single(theta_target.squeeze(), info, s_next_h)
                            q_val_next = Q_tgt[a_best_next, torch.arange(cfg.batch_size, device=cfg.device)].to(DTYPE)
                            
                            # 🎯 수정됨: N-step 사용 여부에 따라 곱해질 감마값을 결정합니다. (n=3이면 gamma^3)
                            target_gamma = (cfg.gamma ** cfg.n_step_size) if cfg.use_n_step else cfg.gamma
                            
                            # 3. 타겟 Y (z_measured) 확정 후 리스트에 얼림 (cfg.gamma 대신 target_gamma 사용)
                            z_measured = (batch_h['r'] + target_gamma * (1 - batch_h['term']) * q_val_next).view(-1, 1)
                            frozen_targets.append(z_measured)
                            
                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            s_flip = batch_hist[0]['s'].t()
                            if normalizer: s_flip = normalizer.normalize(s_flip)
                            argmax_before = forward_single(theta.squeeze(), info, s_flip).argmax(dim=0)

                    for h in range(cfg.N_horizon):
                        theta_before_h = theta.squeeze().clone()
                        
                        # 🎯 아까 밖에서 만들어둔 고정 타겟을 꺼내옵니다.
                        frozen_z = frozen_targets[h]
                        
                        if is_fv:
                            theta, filter_S_info, l_val, t_var, t_k_gain, dbg = srrhuif_step_fv(
                                theta, theta_target, filter_S_info, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache, frozen_z) # 👈 여기 변경
                        else:
                            theta, filter_S_info, l_val, t_var, t_k_gain, dbg = srrhuif_step(
                                theta, theta_target, filter_S_info, batch_hist[h], sp,
                                (h == 0), cfg.p_init, f_cache, frozen_z) # 👈 여기 변경
                        
                        h_delta = theta.squeeze() - theta_before_h
                        if prev_h_delta is not None:
                            d_norm, p_norm = torch.norm(h_delta), torch.norm(prev_h_delta)
                            cos = F.cosine_similarity(h_delta.unsqueeze(0), prev_h_delta.unsqueeze(0)).item() if (d_norm > 1e-8 and p_norm > 1e-8) else 0.0
                            h_cos_traj.append(cos)
                        prev_h_delta = h_delta.clone()
                        
                        ep_l.append(l_val); ep_var.append(t_var); ep_k_gain.append(t_k_gain)
                        ep_i_mean.append(dbg['innov_mean']); ep_i_max.append(dbg['innov_max'])
                        
                        h_k_traj.append(t_k_gain); h_p_traj.append(dbg['avg_P']); h_ht_traj.append(dbg['ht_norm'])
                        h_resid_traj.append(dbg['resid_norm']); h_resid_in_innov_traj.append(dbg['resid_in_innov'])
                        h_ht_theta_traj.append(dbg['ht_theta_in_innov']); h_innov_traj.append(dbg['innov_norm'])
                        h_layer_ht.append(dbg['per_layer_ht']); h_layer_delta.append(dbg['per_layer_delta'])
                        h_layer_resid_max.append(dbg['per_layer_resid_max']) 
                        h_layer_cond.append(dbg['per_layer_cond']); h_layer_ymax.append(dbg['per_layer_ymax'])
                        
                    if cfg.diag_argmax_flip:
                        with torch.no_grad():
                            argmax_after = forward_single(theta.squeeze(), info, s_flip).argmax(dim=0)
                            ep_argmax_flips.append((argmax_before != argmax_after).float().mean().item())
                    
                    if h_layer_cond:
                        for k, v in h_layer_cond[-1].items(): ep_cond_collect.setdefault(k, []).append(v)
                        for k, v in h_layer_ymax[-1].items(): ep_ymax_collect.setdefault(k, []).append(v)

                    theta_target = (1.0 - cfg.tau_srrhuif) * theta_target + cfg.tau_srrhuif * theta
                    last_h_k_traj, last_h_p_traj, last_h_ht_traj = h_k_traj, h_p_traj, h_ht_traj
                    last_h_resid_traj, last_h_cos_traj = h_resid_traj, h_cos_traj
                    last_h_innov_decomp = list(zip(h_resid_in_innov_traj, h_ht_theta_traj, h_innov_traj))
                    last_h_layer_ht, last_h_layer_delta = h_layer_ht, h_layer_delta
                    last_h_layer_resid_max = h_layer_resid_max 
                    last_h_layer_cond, last_h_layer_ymax = h_layer_cond, h_layer_ymax
                    
                update_times.append(time.perf_counter() - update_start) 

            if done or trunc: break

        avg_l = np.mean(ep_l) if ep_l else 0
        avg_v = np.mean(ep_var) if ep_var else 0 
        avg_k = np.mean(ep_k_gain) if ep_k_gain else 0 
        avg_q0 = np.mean(ep_q0) if ep_q0 else 0
        avg_q1 = np.mean(ep_q1) if ep_q1 else 0
        avg_i_mean = np.mean(ep_i_mean) if ep_i_mean else 0
        max_i_max = np.max(ep_i_max) if ep_i_max else 0
        
        logger.add(ep_r, avg_l, cfg.p_init, avg_v, avg_k, avg_q0, avg_q1)
        theta_norms = compute_layer_theta_norms(theta, info)
        null_ratio, null_abs, signal_abs = compute_advantage_null_ratio(theta, info)
        
        eff_rank_val, stable_rank_val = -1.0, -1.0
        if cfg.diag_eff_rank and buffer.current_size >= 128:
            with torch.no_grad():
                diag_batch = buffer.sample_batch(min(256, buffer.current_size))
                s_diag = normalizer.normalize(diag_batch['s'].t()) if normalizer else diag_batch['s'].t()
                _, shared_out = forward_single_with_shared(theta.squeeze(), info, s_diag)
                eff_rank_val, stable_rank_val = compute_effective_rank(shared_out)
        
        avg_cond_dict = {k: float(np.mean(v)) for k, v in ep_cond_collect.items()}
        avg_ymax_dict = {k: float(np.mean(v)) for k, v in ep_ymax_collect.items()}
        avg_argmax_flip = float(np.mean(ep_argmax_flips)) if ep_argmax_flips else 0.0
        
        ref_q = compute_ref_q_values(theta, info, normalizer, cfg.device) if cfg.diag_ref_states else None
        logger.add_diagnostics(avg_cond_dict, avg_ymax_dict, theta_norms, null_ratio, eff_rank_val, stable_rank_val, avg_argmax_flip, ref_q)
        
        buf_info = compute_buffer_diversity(buffer) if cfg.diag_buffer else None
        logger.add_buffer_diag(buf_info, ep)
        just_saturated = buf_info is not None and buf_info['is_saturated'] and not prev_buf_saturated
        if just_saturated: prev_buf_saturated = True
        
        ep_delta = theta.squeeze() - theta_ep_start
        ep_delta_norm = torch.norm(ep_delta).item()
        if prev_ep_delta is not None and ep_delta_norm > 1e-8 and torch.norm(prev_ep_delta) > 1e-8:
            last_ep_cos = F.cosine_similarity(ep_delta.unsqueeze(0), prev_ep_delta.unsqueeze(0)).item()
        else: last_ep_cos = None
        prev_ep_delta = ep_delta.clone()
        target_drift = torch.norm(theta_target.squeeze() - theta.squeeze()).item()

        if ep % cfg.plot_interval == 0: logger.refresh()
        
        if ep % cfg.log_interval == 0:
            recent = np.mean(logger.rewards[-20:]) if len(logger.rewards) >= 20 else np.mean(logger.rewards)
            sat_marker = " 🔔BUF_SATURATED" if just_saturated else ""
            
            print(f"[SRRHUIF] Ep {ep:3d} | Rwd: {ep_r:6.1f} | Avg20: {recent:6.1f} | eps: {eps:.2f} | Buf: {buffer.current_size}/{cfg.buffer_size}{sat_marker} "
                  f"| Loss: {avg_l:.4f} | T_Var: {avg_v:.4f} | Q_std: {cfg.q_init:.1e} | P: {cfg.p_init:.4f} | K_Gain: {avg_k:.4f} "
                  f"| Q(0): {avg_q0:.2f} | Q(1): {avg_q1:.2f} | Time: {time.time()-ep_start:.2f}s")

            file_print(f"          └─▶ Innov (Mean / Max): [{avg_i_mean:.4f} / {max_i_max:.4f}]")
            
            if last_h_k_traj:
                fmt = lambda traj: "[" + ", ".join([f"{v:.4f}" for v in traj]) + "]"
                fmt_e = lambda traj: "[" + ", ".join([f"{v:.2e}" for v in traj]) + "]"
                fmt2 = lambda traj: "[" + ", ".join([f"{v:+.3f}" for v in traj]) + "]"
                file_print(f"          └─▶ K_Gain/h:  {fmt(last_h_k_traj)}")
                file_print(f"          └─▶ P_avg/h:   {fmt_e(last_h_p_traj)}")
                if last_h_innov_decomp:
                    file_print(f"          └─▶ |z-ẑ|/h:   {fmt([d[0] for d in last_h_innov_decomp])}")
                    file_print(f"          └─▶ |H^Tθ|/h:  {fmt([d[1] for d in last_h_innov_decomp])}")
                    file_print(f"          └─▶ |innov|/h: {fmt([d[2] for d in last_h_innov_decomp])}")
                if last_h_cos_traj:
                    file_print(f"          └─▶ cos(δ)/h:  {fmt2(last_h_cos_traj)}")
                ep_cos_str = f"{last_ep_cos:+.3f}" if last_ep_cos is not None else "N/A"
                file_print(f"          └─▶ ep_cos: {ep_cos_str} | θ-target drift: {target_drift:.4f} | ep_Δθ: {ep_delta_norm:.4f}")
                
                if last_h_layer_ht:
                    file_print(f"          ══ [Tier 1] Layer-wise Diagnostics ══")
                    labels = sorted(last_h_layer_ht[0].keys())
                    for h_idx in range(len(last_h_layer_ht)):
                        file_print(f"          ├─▶ ||H^T|| h={h_idx}:  " + " ".join([f"{l}={last_h_layer_ht[h_idx][l]:.2f}" for l in labels]))
                    for h_idx in range(len(last_h_layer_resid_max)):
                        file_print(f"          ├─▶ ResMax  h={h_idx}:  " + " ".join([f"{l}={last_h_layer_resid_max[h_idx][l]:.2f}" for l in labels]))
                    for h_idx in range(len(last_h_layer_delta)):
                        file_print(f"          ├─▶ ||Δθ||  h={h_idx}:  " + " ".join([f"{l}={last_h_layer_delta[h_idx][l]:.4f}" for l in labels]))
                    max_ht_per_layer = {l: max(last_h_layer_ht[h][l] for h in range(len(last_h_layer_ht))) for l in labels}
                    dominant = max(max_ht_per_layer, key=max_ht_per_layer.get)
                    file_print(f"          └─▶ Dominant layer: {dominant} (max||H^T||={max_ht_per_layer[dominant]:.1f})")
            
            file_print(f"          ══ DIAGNOSTICS ══")
            if last_h_layer_cond:
                labels = sorted(last_h_layer_cond[0].keys())
                for h_idx in range(len(last_h_layer_cond)):
                    file_print(f"          ├─▶ cond(Y)h={h_idx}: " + " ".join([f"{l}={last_h_layer_cond[h_idx][l]:.1e}" for l in labels]))
                for h_idx in range(len(last_h_layer_ymax)):
                    file_print(f"          ├─▶ Y_max  h={h_idx}: " + " ".join([f"{l}={last_h_layer_ymax[h_idx][l]:.1e}" for l in labels]))
            
            labels = sorted(theta_norms.keys())
            file_print(f"          ├─▶ ||θ|| per layer:   " + " ".join([f"{l}={theta_norms[l]:.3f}" for l in labels]))
            file_print(f"          ├─▶ Adv/Q null/signal: ratio={null_ratio:.4f} (null={null_abs:.4f}, signal={signal_abs:.4f})")
            if eff_rank_val > 0:
                file_print(f"          ├─▶ Shared rank:       eff={eff_rank_val:.1f}/{cfg.shared_layers[-1]}, stable={stable_rank_val:.2f}")
            file_print(f"          ├─▶ Argmax flip rate:  {avg_argmax_flip:.4f} (updates={len(ep_argmax_flips)})")
            if buf_info is not None:
                sat_str = "YES" if buf_info['is_saturated'] else "no"
                file_print(f"          ├─▶ Buffer diag:       fill={buf_info['fill_ratio']:.3f}({sat_str}) state_std={buf_info['state_std']:.4f} state_range={buf_info['state_range']:.3f}")
                file_print(f"          ├─▶ Buffer samples:    done_ratio={buf_info['done_ratio']:.4f} r_std={buf_info['reward_std']:.4f} r_mean={buf_info['reward_mean']:.4f}")
                file_print(f"          ├─▶ Buffer age:        ep[{buf_info['age_min']}..{buf_info['age_max']}] range={buf_info['age_range']} std={buf_info['age_std']:.2f}")
            if ref_q:
                file_print(f"          └─▶ Ref states:        " + " ".join([f"{name}:ΔQ={ref_q[name]['dq']:+.4f}(a={ref_q[name]['argmax']})" for name in REF_NAMES]))

    logger.total_time = time.time() - train_start_time
    logger.avg_step_time = (np.mean(update_times) * 1000) if update_times else 0.0 
    env.close()
    logger.refresh()
    logger.save_diagnostic_plots()
    
    try:
        plot_cartpole_state_landscape(theta, info, cfg, normalizer, method_title, cfg.param_str)
    except Exception as e: print(f"[경고] 지형도 생성 중 오류 발생: {e}")
        
    logger.close()

if __name__ == "__main__":
    if cfg.save_file_log: setup_file_logging(os.path.join(cfg.outdir, "training_log.txt"))
    try: train_srrhuif()
    finally:
        if cfg.save_file_log: close_file_logging()