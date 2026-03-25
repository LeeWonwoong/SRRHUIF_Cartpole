function main_CartPole_debug_ND_EG()
%% =========================================================================
% SRRHUIF Node Decoupled (ND) 발산 분석용 확장 디버깅 코드 (Epsilon-Greedy)
% =========================================================================
clc; close all;

%% 1. 환경 설정
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
dimS = prod(obsInfo.Dimension);
nA   = numel(actInfo.Elements);
actElements = actInfo.Elements;

%% 2. 파라미터 설정
param.M = 4; 
param.shared_layers = [16, 16];
param.value_layers = [4];
param.advantage_layers = [4];

maxEpisodes = 200;
maxSteps    = 500; 
batchSz     = 128;           

gamma       = 0.94;         
sp.r_measure_std = 2.0;    % ★ ND용 고정 R
sp.scale_factor = 1;       

epsHi = 1.0; 
epsLo = 0.001; 
EPS_DECAY = 3000;
tau_soft = 0.005;

sp.N_horizon = 4;
sp.P_min = 0.001;
sp.P_max = 0.01;
sp.adaptive_window = 12;
sp.max_score = maxSteps;
sp.use_adaptive_P = true;
sp.q_process_std = 5e-3;   % ★ Process Q 유지

% ★ ND 필수 세팅
sp.alpha = 0.99;           % alpha=0.99로 파라미터 붕괴 방지
sp.beta  = 2;
sp.kappa = 0;
bufMax = 100000;

[DuelingInfo, nd_layers] = create_dueling_layer_info(dimS, nA, param.shared_layers, ...
                                         param.value_layers, param.advantage_layers);
n_x = DuelingInfo.total_params;
sp.DuelingInfo = DuelingInfo;
sp.nd_layers = nd_layers;
sp.num_nd_layers = length(nd_layers);

fprintf('==========================================================\n');
fprintf('  SRRHUIF ND Extended Debug Mode: Divergence Analysis\n');
fprintf('==========================================================\n');
fprintf('gamma=%.2f | R=%.2f | scale_factor=%d | Params=%d\n', ...
        gamma, sp.r_measure_std, sp.scale_factor, n_x);
fprintf('Expected Q_max ≈ %.1f | SNR(Q/R) ≈ %.1f\n', ...
        (1/sp.scale_factor)/(1-gamma), (1/sp.scale_factor)/(1-gamma)/sp.r_measure_std);
fprintf('==========================================================\n');

%% 3. 가중치 초기화
rng(42);
theta_init = initialize_dueling_weights(DuelingInfo);

%% 4. SRRHUIF 파라미터 설정
sp.gamma = gamma;
sp.n_x = n_x;
sp.batchSz = batchSz;

eps_param.epsHi = epsHi;
eps_param.epsLo = epsLo;
eps_param.EPS_DECAY = EPS_DECAY;

%% 5. 실시간 플롯 설정 (3x4 = 12개 subplot)
fig = figure('Name', 'SRRHUIF ND Extended Debug', 'Color', 'w', 'Position', [30 30 1700 900]);

% ===== Row 1: 기본 학습 지표 =====
subplot(3,4,1);
h_reward_raw = animatedline('Color', [0.3 0.3 1 0.3], 'LineWidth', 0.5);
h_reward_ma = animatedline('Color', 'b', 'LineWidth', 2);
yline(195, 'g--'); yline(500, 'k--');
xlabel('Episode'); ylabel('Reward'); title('1. Episode Reward');
xlim([1 maxEpisodes]); ylim([0 550]); grid on;

subplot(3,4,2);
h_loss = animatedline('Color', 'r', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Loss'); title('2. TD Loss (MA20)');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,3);
h_residual_mean = animatedline('Color', 'b', 'LineWidth', 1.5);
h_residual_max = animatedline('Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
xlabel('Episode'); ylabel('Value'); title('3. Node Residual (Mean/Max)');
legend('Mean|r|', 'Max|r|', 'Location', 'best');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,4);
h_innov_mean = animatedline('Color', 'b', 'LineWidth', 1.5);
h_innov_max = animatedline('Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
xlabel('Episode'); ylabel('Value'); title('4. Node Innovation (Mean/Max)');
legend('Mean|i|', 'Max|i|', 'Location', 'best');
xlim([1 maxEpisodes]); grid on;

% ===== Row 2: 정보 갱신 분석 =====
subplot(3,4,5);
h_gain_mean = animatedline('Color', 'b', 'LineWidth', 1.5);
h_gain_max = animatedline('Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
xlabel('Episode'); ylabel('Value'); title('5. Node Update Gain (Mean/Max)');
legend('Mean', 'Max', 'Location', 'best');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,6);
h_delta_y = animatedline('Color', [0.5 0 0.5], 'LineWidth', 1.5);
xlabel('Episode'); ylabel('||\Deltay||_mean'); title('6. Node Info Vector Change');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,7);
h_delta_theta = animatedline('Color', [0 0.5 0], 'LineWidth', 1.5);
yline(0.1, 'g--', 'LineWidth', 1);  yline(0.2, 'y--', 'LineWidth', 1);  yline(0.3, 'r--', 'LineWidth', 1);
xlabel('Episode'); ylabel('||\Delta\theta||_mean'); title('7. Node Parameter Change ★');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,8);
h_q_scale = animatedline('Color', [0.8 0.4 0], 'LineWidth', 1.5);
yline((1/sp.scale_factor) / (1-gamma), 'k--', 'LineWidth', 1); 
xlabel('Episode'); ylabel('|Q|'); title(sprintf('8. Q Scale (theory: %.1f)', (1/sp.scale_factor) / (1-gamma)));
xlim([1 maxEpisodes]); grid on;

% ===== Row 3: 행렬/파라미터 건강 상태 =====
subplot(3,4,9);
h_H_norm = animatedline('Color', [0.5 0 0.5], 'LineWidth', 1.5);
xlabel('Episode'); ylabel('||H||_F_mean'); title('9. Node Obs Matrix ||H||');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,10);
h_cond_Y = animatedline('Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
yline(8, 'y--', 'LineWidth', 1); yline(10, 'r--', 'LineWidth', 1); 
xlabel('Episode'); ylabel('log_{10}(cond_max)'); title('10. Max Node cond(Y) - Log');
xlim([1 maxEpisodes]); ylim([0 15]); grid on;

subplot(3,4,11);
h_snr = animatedline('Color', [0.8 0.4 0], 'LineWidth', 1.5);
yline(3.0, 'g--', 'LineWidth', 1); yline(5.0, 'y--', 'LineWidth', 1); yline(10.0, 'r--', 'LineWidth', 1);
xlabel('Episode'); ylabel('SNR (Q/R)'); title('11. SNR (Q/R) - Early Warning ★');
xlim([1 maxEpisodes]); grid on;

subplot(3,4,12);
h_p_init = animatedline('Color', [0 0.6 0], 'LineWidth', 1.5);
yline(sp.P_min, 'b--', 'LineWidth', 1); yline(sp.P_max, 'r--', 'LineWidth', 1);
xlabel('Episode'); ylabel('P_{init}'); title('12. Adaptive P_{init}');
xlim([1 maxEpisodes]); ylim([0 sp.P_max*1.2]); grid on;

sgtitle(sprintf('ND EG: \\gamma=%.2f, R=%.2f, scale=%d, SNR=%.2f', ...
        gamma, sp.r_measure_std, sp.scale_factor, (1/sp.scale_factor)/(1-gamma)/sp.r_measure_std), ...
        'FontSize', 14, 'FontWeight', 'bold');
drawnow;

%% 6. 학습 루프
scale_factors = [2.4; 3.0; 0.21; 2.0];
theta = theta_init; theta_target = theta_init;

B.S = zeros(dimS, bufMax); B.A = zeros(1, bufMax, 'uint8');
B.R = zeros(1, bufMax); B.S_Next = zeros(dimS, bufMax); B.term = false(1, bufMax);
bufCnt = 0; batch_history = cell(sp.N_horizon, 1); steps_done = 0;

reward_history = zeros(maxEpisodes, 1); loss_history = zeros(maxEpisodes, 1);
residual_mean_hist = zeros(maxEpisodes, 1); residual_max_hist = zeros(maxEpisodes, 1);
innov_mean_hist = zeros(maxEpisodes, 1); innov_max_hist = zeros(maxEpisodes, 1);
gain_mean_hist = zeros(maxEpisodes, 1); gain_max_hist = zeros(maxEpisodes, 1);
delta_y_hist = zeros(maxEpisodes, 1); delta_theta_hist = zeros(maxEpisodes, 1);
q_scale_hist = zeros(maxEpisodes, 1); H_norm_hist = zeros(maxEpisodes, 1);
cond_Y_hist = zeros(maxEpisodes, 1); snr_hist = zeros(maxEpisodes, 1);
p_init_hist = zeros(maxEpisodes, 1);

current_P_init = sp.P_max;

% ND 캐시 초기화
neuron_S_info = cell(sp.num_nd_layers, 1);
for L = 1:sp.num_nd_layers
    fan_out = nd_layers(L).W_rows; n_per = nd_layers(L).n_per;
    neuron_S_info{L} = repmat((1/sqrt(sp.P_max)) * eye(n_per), [1, 1, fan_out]);
end

for ep = 1:maxEpisodes
    obs = reset(env); sVec = double(obs(:)) ./ scale_factors;
    epReward = 0; epLossSum = 0; lossCount = 0;
    
    ep_residual_mean = []; ep_residual_max = [];
    ep_innov_mean = []; ep_innov_max = [];
    ep_gain_mean = []; ep_gain_max = [];
    ep_delta_y = []; ep_delta_theta = []; ep_q_scale = [];
    ep_H_norm = []; ep_cond_Y = [];
    
    for t = 1:maxSteps
        epsVal = eps_param.epsLo + (eps_param.epsHi - eps_param.epsLo) * exp(-steps_done / eps_param.EPS_DECAY);
        steps_done = steps_done + 1;
        
        qPred = Dueling_forward_vec(theta, DuelingInfo, sVec);
        if rand < epsVal, aIdx = randi(nA); else, [~, aIdx] = max(qPred); end
        
        aForce = actElements(aIdx);
        [nextObs, reward, isDone, ~] = step(env, aForce);
        sNextVec = double(nextObs(:)) ./ scale_factors;

        idx = mod(bufCnt, bufMax) + 1; bufCnt = bufCnt + 1;
        B.S(:, idx) = sVec; B.A(idx) = aIdx; B.R(idx) = reward / sp.scale_factor;
        B.S_Next(:, idx) = sNextVec; B.term(idx) = isDone;
        
        if bufCnt >= batchSz
            currentBufSize = min(bufCnt, bufMax);
            new_batch = randperm(currentBufSize, batchSz);
            batch_history = [batch_history(2:end); {new_batch}];
            
            if all(~cellfun(@isempty, batch_history))
                sp.current_P_init = current_P_init;
                
                % ★ ND Horizon Update (확장 디버그 포함)
                [theta, neuron_S_info, curr_loss, debug_stats] = run_srrhuif_horizon_debug_nd(...
                    theta, theta_target, neuron_S_info, batch_history, B, sp);
                
                ep_residual_mean(end+1) = debug_stats.residual_mean; ep_residual_max(end+1) = debug_stats.residual_max;
                ep_innov_mean(end+1) = debug_stats.innov_mean; ep_innov_max(end+1) = debug_stats.innov_max;
                ep_gain_mean(end+1) = debug_stats.gain_mean; ep_gain_max(end+1) = debug_stats.gain_max;
                ep_delta_y(end+1) = debug_stats.delta_y; ep_delta_theta(end+1) = debug_stats.delta_theta;
                ep_q_scale(end+1) = debug_stats.q_scale; ep_H_norm(end+1) = debug_stats.H_norm;
                ep_cond_Y(end+1) = debug_stats.cond_Y_max; % ★ 가장 나쁜 뉴런의 cond
                
                theta_target = (1 - tau_soft) * theta_target + tau_soft * theta;
                epLossSum = epLossSum + curr_loss; lossCount = lossCount + 1;
            end
        end
        epReward = epReward + reward; sVec = sNextVec;
        if isDone, break; end
    end
    
    reward_history(ep) = epReward; loss_history(ep) = epLossSum / max(lossCount, 1);
    
    if ~isempty(ep_residual_mean)
        residual_mean_hist(ep) = mean(ep_residual_mean); residual_max_hist(ep) = max(ep_residual_max);
        innov_mean_hist(ep) = mean(ep_innov_mean); innov_max_hist(ep) = max(ep_innov_max);
        gain_mean_hist(ep) = mean(ep_gain_mean); gain_max_hist(ep) = max(ep_gain_max);
        delta_y_hist(ep) = mean(ep_delta_y); delta_theta_hist(ep) = mean(ep_delta_theta);
        q_scale_hist(ep) = mean(ep_q_scale); H_norm_hist(ep) = mean(ep_H_norm);
        cond_Y_hist(ep) = max(ep_cond_Y); % 에피소드 중 가장 컸던 노드의 cond
        snr_hist(ep) = q_scale_hist(ep) / sp.r_measure_std;
    end
    
    p_init_hist(ep) = current_P_init;
    if sp.use_adaptive_P && ep >= 1, current_P_init = compute_adaptive_P_init(reward_history, ep, sp); end
    
    % 플롯 추가
    addpoints(h_reward_raw, ep, epReward);
    if ep >= 20
        addpoints(h_reward_ma, ep, mean(reward_history(ep-19:ep)));
        addpoints(h_loss, ep, mean(loss_history(max(1,ep-19):ep)));
    end
    addpoints(h_residual_mean, ep, residual_mean_hist(ep)); addpoints(h_residual_max, ep, residual_max_hist(ep));
    addpoints(h_innov_mean, ep, innov_mean_hist(ep)); addpoints(h_innov_max, ep, innov_max_hist(ep));
    addpoints(h_gain_mean, ep, gain_mean_hist(ep)); addpoints(h_gain_max, ep, gain_max_hist(ep));
    addpoints(h_delta_y, ep, delta_y_hist(ep)); addpoints(h_delta_theta, ep, delta_theta_hist(ep));
    addpoints(h_q_scale, ep, q_scale_hist(ep)); addpoints(h_H_norm, ep, H_norm_hist(ep));
    addpoints(h_cond_Y, ep, log10(max(cond_Y_hist(ep), 1))); addpoints(h_snr, ep, snr_hist(ep));
    addpoints(h_p_init, ep, p_init_hist(ep));
    
    if mod(ep, 20) == 0
        ma20 = mean(reward_history(max(1,ep-19):ep));
        fprintf('Ep %3d | R: %5.1f | MA20: %5.1f | Loss: %.2f | ND_Δθ: %.3f | Max_cond: %.1e | SNR: %.2f\n', ...
                ep, epReward, ma20, loss_history(ep), delta_theta_hist(ep), cond_Y_hist(ep), snr_hist(ep));
        drawnow limitrate;
    end
end

%% 7. 최종 분석 리포트
fprintf('\n══════════════════════════════════════════════════════════\n');
fprintf('  발산 분석 최종 리포트 (Node Decoupled)\n');
fprintf('══════════════════════════════════════════════════════════\n');
fprintf('\n[설정]\n  gamma = %.2f\n  R = %.2f\n', gamma, sp.r_measure_std);

[max_gain, max_gain_ep] = max(gain_max_hist);
[max_dtheta, max_dtheta_ep] = max(delta_theta_hist);
[max_cond, max_cond_ep] = max(cond_Y_hist);
[max_residual, max_residual_ep] = max(residual_max_hist);

fprintf('\n[최대값 분석]\n');
fprintf('  최대 Residual (전체 노드 중): %.2f (Episode %d)\n', max_residual, max_residual_ep);
fprintf('  최대 Update Gain (전체 노드 중): %.2f (Episode %d)\n', max_gain, max_gain_ep);
fprintf('  평균 노드 Δθ 최대치: %.3f (Episode %d)\n', max_dtheta, max_dtheta_ep);
fprintf('  가장 불안정했던 노드의 cond(Y): %.2e (Episode %d)\n', max_cond, max_cond_ep);

avg_q = mean(q_scale_hist(q_scale_hist > 0)); measured_snr = avg_q / sp.r_measure_std;
fprintf('\n[SNR 분석]\n  평균 Q Scale: %.2f\n  측정 SNR (Q/R): %.2f\n', avg_q, measured_snr);
fprintf('══════════════════════════════════════════════════════════\n');

end

%% ========================================================================
%  SRRHUIF Horizon Update (ND + Debug)
%% ========================================================================
function [theta_new, neuron_S_info_new, avg_loss, debug_stats] = run_srrhuif_horizon_debug_nd(theta, theta_target, neuron_S_info, batch_history, B, sp)
    N_horizon = sp.N_horizon;
    if isfield(sp, 'current_P_init'), initial_P_std = sp.current_P_init; else, initial_P_std = sp.P_max; end
    
    total_loss = 0;
    
    all_res_abs = []; all_inn_abs = []; all_gain = [];
    all_dy = []; all_dtheta = []; all_H = []; all_cond = [];
    all_q_scale = [];
    
    for h = 1:N_horizon
        batchIdx = batch_history{h}; is_first_step = (h == 1);
        [theta, neuron_S_info, loss_step, step_debug] = srrhuif_step_nd_debug(...
            theta, theta_target, neuron_S_info, B, batchIdx, sp, is_first_step, initial_P_std);
        total_loss = total_loss + loss_step;
        
        all_res_abs = [all_res_abs; step_debug.res_abs_all];
        all_inn_abs = [all_inn_abs; step_debug.inn_abs_all];
        all_gain = [all_gain; step_debug.gain_all];
        all_dy = [all_dy; step_debug.dy_all];
        all_dtheta = [all_dtheta; step_debug.dtheta_all];
        all_H = [all_H; step_debug.H_all];
        all_cond = [all_cond; step_debug.cond_all];
        all_q_scale(end+1) = step_debug.q_scale;
    end
    
    theta_new = theta; neuron_S_info_new = neuron_S_info; avg_loss = total_loss / N_horizon;
    
    % 노드 1개당 평균 및 전체 노드 중 가장 튄 Max 값 계산
    debug_stats.residual_mean = mean(all_res_abs); debug_stats.residual_max = max(all_res_abs);
    debug_stats.innov_mean = mean(all_inn_abs); debug_stats.innov_max = max(all_inn_abs);
    debug_stats.gain_mean = mean(all_gain); debug_stats.gain_max = max(all_gain);
    debug_stats.delta_y = mean(all_dy); debug_stats.delta_theta = mean(all_dtheta);
    debug_stats.H_norm = mean(all_H); debug_stats.cond_Y_max = max(all_cond);
    debug_stats.q_scale = mean(all_q_scale);
end

%% ========================================================================
%  SRRHUIF ND Single Step (Q 유지, 고정 R, Debug)
%% ========================================================================
%% [수정된 ND Single Step] Pure Parallel (Jacobi) 방식
function [theta_new_out, new_neuron_S_info, avg_loss, step_debug] = srrhuif_step_nd_debug(theta_current_in, theta_target, neuron_S_info, B, batchIdx, sp, is_first_step, p_init_val)
    % 모든 노드가 공유할 고정된 베이스 파라미터 (Jacobi 방식의 핵심)
    theta_base = theta_current_in; 
    theta_prior_for_S = theta_target; if ~is_first_step, theta_prior_for_S = theta_current_in; end
    
    % 업데이트 결과를 담을 임시 버퍼 (마지막에 한꺼번에 반영)
    theta_new_out = theta_current_in; 
    
    gamma_discount = sp.gamma; batchSz = length(batchIdx);
    States_Batch = B.S(:, batchIdx); Actions_Batch = double(B.A(batchIdx));
    Rewards_Batch = B.R(batchIdx)'; S_Next_Batch = B.S_Next(:, batchIdx); Term_Batch = double(B.term(batchIdx)');
    
    new_neuron_S_info = cell(sp.num_nd_layers, 1);
    total_loss = 0; node_count = 0;
    res_abs_all = []; inn_abs_all = []; gain_all = []; dy_all = []; dtheta_all = []; H_all = []; cond_all = [];
    
    % Target 값 계산 (모든 노드가 동일한 Target 사용)
    Q_online_next = Dueling_forward_vec_batch(theta_base, sp.DuelingInfo, S_Next_Batch);
    [~, a_best_next] = max(Q_online_next, [], 1);
    Q_target_next = Dueling_forward_vec_batch(theta_target, sp.DuelingInfo, S_Next_Batch);
    idx_ddqn = sub2ind(size(Q_target_next), a_best_next, 1:batchSz);
    z_measured = Rewards_Batch + gamma_discount * (1 - Term_Batch) .* Q_target_next(idx_ddqn)';

    for L = 1:sp.num_nd_layers
        fan_in = sp.nd_layers(L).W_cols; fan_out = sp.nd_layers(L).W_rows; n_per = sp.nd_layers(L).n_per;
        W_start = sp.nd_layers(L).W_start; b_start = sp.nd_layers(L).b_start;
        [Wm, Wc, gamma_sigma] = compute_ut_weights(n_per, sp.alpha, sp.beta, sp.kappa);
        new_S_L = zeros(n_per, n_per, fan_out);
        
        for f = 1:fan_out
            w_idx = W_start + (f-1)*fan_in : W_start + f*fan_in - 1; b_idx = b_start + f - 1;
            % 1. Time Update (동일한 theta_prior_for_S 기준)
            if is_first_step
                P_sqrt_prev = sqrt(p_init_val) * eye(n_per);
            else
                P_sqrt_prev = safe_inv_tril(neuron_S_info{L}(:,:,f), n_per);
            end
            S_Q = sp.q_process_std * eye(n_per);
            P_sqrt_pred = tria_operation([P_sqrt_prev, S_Q]);
            S_pred = safe_inv_tril(P_sqrt_pred, n_per);
            
            theta_node_prior = [theta_prior_for_S(w_idx); theta_prior_for_S(b_idx)];
            y_pred = S_pred * (S_pred' * theta_node_prior);
            
            % 2. Sigma Points & Forward (★중요: 항상 고정된 theta_base를 바탕으로 생성)
            X_sigma = zeros(n_per, 2*n_per + 1); X_sigma(:, 1) = theta_node_prior;
            scaled_P = gamma_sigma * P_sqrt_pred;
            for k = 1:n_per
                X_sigma(:, k+1)       = theta_node_prior + scaled_P(:, k);
                X_sigma(:, n_per+k+1) = theta_node_prior - scaled_P(:, k);
            end
            
            Z_sigma = zeros(batchSz, 2*n_per + 1);
            for s = 1:(2*n_per + 1)
                % 노드 f 업데이트 전의 theta_base를 복사해서 사용 (Jacobi)
                theta_temp = theta_base; 
                theta_temp(w_idx) = X_sigma(1:fan_in, s);
                theta_temp(b_idx) = X_sigma(end, s);
                
                Q_all = Dueling_forward_vec_batch(theta_temp, sp.DuelingInfo, States_Batch);
                Z_sigma(:, s) = Q_all(sub2ind(size(Q_all), Actions_Batch, 1:batchSz))';
            end
            
            z_hat = Z_sigma * Wm';
            residual = z_measured - z_hat;
            P_xz = ((X_sigma - theta_node_prior) .* Wc) * (Z_sigma - z_hat)';
            HT = S_pred * (S_pred' * P_xz);
            
            % 3. Measurement Update
            S_new = tria_operation([S_pred, HT * (1/sp.r_measure_std)]);
            innovation = residual + HT' * theta_node_prior;
            update_gain = HT * ((1/sp.r_measure_std^2) * innovation);
            
            theta_node_new = robust_solve_spd(S_new, y_pred + update_gain, n_per);
            
            % ★ 즉시 반영하지 않고 임시 버퍼(theta_new_out)에 저장
            theta_new_out(w_idx) = theta_node_new(1:fan_in);
            theta_new_out(b_idx) = theta_node_new(end);
            
            new_S_L(:,:,f) = S_new;
            
            % 디버그 통계 수집
            res_abs_all = [res_abs_all; abs(residual)]; inn_abs_all = [inn_abs_all; abs(innovation)];
            gain_all(end+1) = norm(update_gain); dy_all(end+1) = norm(update_gain);
            dtheta_all(end+1) = norm(theta_node_new - theta_node_prior);
            H_all(end+1) = norm(HT, 'fro');
            Y_new = S_new * S_new' + 1e-6 * eye(n_per); cond_all(end+1) = cond(Y_new);
            total_loss = total_loss + mean(residual.^2); node_count = node_count + 1;
        end
        new_neuron_S_info{L} = new_S_L;
    end
    avg_loss = total_loss / node_count;
    step_debug.res_abs_all = res_abs_all; step_debug.inn_abs_all = inn_abs_all;
    step_debug.gain_all = gain_all'; step_debug.dy_all = dy_all';
    step_debug.dtheta_all = dtheta_all'; step_debug.H_all = H_all';
    step_debug.cond_all = cond_all'; step_debug.q_scale = mean(abs(z_measured));
end

%% Helper Functions (생략 없이 동일하게 유지됨)
function P_init = compute_adaptive_P_init(reward_history, ep, sp)
    start_idx = max(1, ep - sp.adaptive_window + 1);
    current_score = mean(reward_history(start_idx:ep));
    gap = max(0, min(1, 1 - current_score / sp.max_score));
    P_init = sp.P_min + (sp.P_max - sp.P_min) * gap;
end

function [Wm, Wc, gamma_sigma] = compute_ut_weights(n, alpha, beta, kappa)
    lambda_val = alpha^2 * (n + kappa) - n;
    gamma_sigma = sqrt(n + lambda_val);
    Wm = zeros(1, 2*n + 1); Wc = zeros(1, 2*n + 1);
    Wm(1) = lambda_val / (n + lambda_val);
    Wc(1) = Wm(1) + (1 - alpha^2 + beta);
    Wm(2:end) = 0.5 / (n + lambda_val);
    Wc(2:end) = 0.5 / (n + lambda_val);
end

function L_inv = safe_inv_tril(L, n)
    opts.LT = true;
    [L_inv, rcond_val] = linsolve(L, eye(n), opts);
    if any(~isfinite(L_inv(:))) || rcond_val < 1e-15
        L_jit = L + 1e-6 * eye(n);
        try L_inv = linsolve(L_jit, eye(n), opts); catch, L_inv = pinv(L_jit); end
    end
end

function theta = robust_solve_spd(S_tril, y, n)
    opts_fwd.LT = true; [z, rcond1] = linsolve(S_tril, y, opts_fwd);
    opts_bwd.UT = true; [theta, rcond2] = linsolve(S_tril', z, opts_bwd);
    if any(~isfinite(theta(:))) || rcond1 < 1e-15 || rcond2 < 1e-15
        Y_reg = S_tril * S_tril' + 1e-6 * eye(n);
        theta = Y_reg \ y;
        if any(~isfinite(theta(:))), theta = pinv(Y_reg) * y; end
    end
end

function S = tria_operation(A)
    n = size(A, 1); [~, R] = qr(A', 0);
    if size(R, 1) < n, R = [R; zeros(n - size(R,1), size(R,2))]; end
    S = R(1:n, 1:n)'; 
    d = diag(S); 
    signs = sign(d); signs(signs == 0) = 1; 
    S = S .* signs';
    
    % ★ Python의 .clamp_(min=1e-6)와 동일한 수치 안정성 보장 로직 추가
    d = diag(S);
    d(d < 1e-6) = 1e-6;
    S(1:n+1:end) = d; 
end

function [DuelingInfo, nd_layers] = create_dueling_layer_info(dimS, nA, shared_layers, value_layers, advantage_layers)
    DuelingInfo.dimS = dimS; DuelingInfo.nA = nA; idx = 1; nd_layers = [];
    function nd_layers = add_to_nd(nd_layers, W_start, W_len, b_start, b_len, fan_in, fan_out)
        s.W_start = W_start; s.W_len = W_len; s.b_start = b_start; s.b_len = b_len;
        s.W_cols = fan_in; s.W_rows = fan_out; s.n_per = fan_in + 1;
        nd_layers = [nd_layers, s];
    end
    shared_sizes = [dimS, shared_layers]; num_shared = length(shared_sizes) - 1;
    DuelingInfo.shared_num_layers = num_shared;
    DuelingInfo.shared_W_start = zeros(num_shared, 1); DuelingInfo.shared_W_len = zeros(num_shared, 1);
    DuelingInfo.shared_W_rows = zeros(num_shared, 1); DuelingInfo.shared_W_cols = zeros(num_shared, 1);
    DuelingInfo.shared_b_start = zeros(num_shared, 1); DuelingInfo.shared_b_len = zeros(num_shared, 1);
    for L = 1:num_shared
        fan_in = shared_sizes(L); fan_out = shared_sizes(L+1);
        DuelingInfo.shared_W_start(L) = idx; DuelingInfo.shared_W_len(L) = fan_out * fan_in;
        DuelingInfo.shared_W_rows(L) = fan_out; DuelingInfo.shared_W_cols(L) = fan_in;
        W_s = idx; W_l = fan_out * fan_in; idx = idx + W_l;
        DuelingInfo.shared_b_start(L) = idx; DuelingInfo.shared_b_len(L) = fan_out;
        b_s = idx; b_l = fan_out; idx = idx + b_l;
        nd_layers = add_to_nd(nd_layers, W_s, W_l, b_s, b_l, fan_in, fan_out);
    end
    shared_output_dim = shared_sizes(end); DuelingInfo.shared_params = idx - 1;
    value_sizes = [shared_output_dim, value_layers, 1]; num_value = length(value_sizes) - 1;
    DuelingInfo.value_num_layers = num_value;
    DuelingInfo.value_W_start = zeros(num_value, 1); DuelingInfo.value_W_len = zeros(num_value, 1);
    DuelingInfo.value_W_rows = zeros(num_value, 1); DuelingInfo.value_W_cols = zeros(num_value, 1);
    DuelingInfo.value_b_start = zeros(num_value, 1); DuelingInfo.value_b_len = zeros(num_value, 1);
    value_start_idx = idx;
    for L = 1:num_value
        fan_in = value_sizes(L); fan_out = value_sizes(L+1);
        DuelingInfo.value_W_start(L) = idx; DuelingInfo.value_W_len(L) = fan_out * fan_in;
        DuelingInfo.value_W_rows(L) = fan_out; DuelingInfo.value_W_cols(L) = fan_in;
        W_s = idx; W_l = fan_out * fan_in; idx = idx + W_l;
        DuelingInfo.value_b_start(L) = idx; DuelingInfo.value_b_len(L) = fan_out;
        b_s = idx; b_l = fan_out; idx = idx + b_l;
        nd_layers = add_to_nd(nd_layers, W_s, W_l, b_s, b_l, fan_in, fan_out);
    end
    DuelingInfo.value_params = idx - value_start_idx;
    advantage_sizes = [shared_output_dim, advantage_layers, nA]; num_advantage = length(advantage_sizes) - 1;
    DuelingInfo.advantage_num_layers = num_advantage;
    DuelingInfo.advantage_W_start = zeros(num_advantage, 1); DuelingInfo.advantage_W_len = zeros(num_advantage, 1);
    DuelingInfo.advantage_W_rows = zeros(num_advantage, 1); DuelingInfo.advantage_W_cols = zeros(num_advantage, 1);
    DuelingInfo.advantage_b_start = zeros(num_advantage, 1); DuelingInfo.advantage_b_len = zeros(num_advantage, 1);
    advantage_start_idx = idx;
    for L = 1:num_advantage
        fan_in = advantage_sizes(L); fan_out = advantage_sizes(L+1);
        DuelingInfo.advantage_W_start(L) = idx; DuelingInfo.advantage_W_len(L) = fan_out * fan_in;
        DuelingInfo.advantage_W_rows(L) = fan_out; DuelingInfo.advantage_W_cols(L) = fan_in;
        W_s = idx; W_l = fan_out * fan_in; idx = idx + W_l;
        DuelingInfo.advantage_b_start(L) = idx; DuelingInfo.advantage_b_len(L) = fan_out;
        b_s = idx; b_l = fan_out; idx = idx + b_l;
        nd_layers = add_to_nd(nd_layers, W_s, W_l, b_s, b_l, fan_in, fan_out);
    end
    DuelingInfo.advantage_params = idx - advantage_start_idx; DuelingInfo.total_params = idx - 1;
end

function theta = initialize_dueling_weights(DuelingInfo)
    n_x = DuelingInfo.total_params; theta = zeros(n_x, 1);
    for L = 1:DuelingInfo.shared_num_layers
        fan_in = DuelingInfo.shared_W_cols(L); W_len = DuelingInfo.shared_W_len(L); W_start = DuelingInfo.shared_W_start(L);
        theta(W_start:W_start+W_len-1) = randn(W_len, 1) * sqrt(2/fan_in);
    end
    for L = 1:DuelingInfo.value_num_layers
        fan_in = DuelingInfo.value_W_cols(L); W_len = DuelingInfo.value_W_len(L); W_start = DuelingInfo.value_W_start(L);
        theta(W_start:W_start+W_len-1) = randn(W_len, 1) * sqrt(2/fan_in);
    end
    for L = 1:DuelingInfo.advantage_num_layers
        fan_in = DuelingInfo.advantage_W_cols(L); W_len = DuelingInfo.advantage_W_len(L); W_start = DuelingInfo.advantage_W_start(L);
        theta(W_start:W_start+W_len-1) = randn(W_len, 1) * sqrt(2/fan_in);
    end
end

function Q = Dueling_forward_vec(theta, DuelingInfo, x)
    A = x;
    for L = 1:DuelingInfo.shared_num_layers
        W = reshape(theta(DuelingInfo.shared_W_start(L):DuelingInfo.shared_W_start(L)+DuelingInfo.shared_W_len(L)-1), DuelingInfo.shared_W_rows(L), DuelingInfo.shared_W_cols(L));
        b = theta(DuelingInfo.shared_b_start(L):DuelingInfo.shared_b_start(L)+DuelingInfo.shared_b_len(L)-1);
        A = max(0, W * A + b);
    end
    shared_output = A; V = shared_output;
    for L = 1:DuelingInfo.value_num_layers
        W = reshape(theta(DuelingInfo.value_W_start(L):DuelingInfo.value_W_start(L)+DuelingInfo.value_W_len(L)-1), DuelingInfo.value_W_rows(L), DuelingInfo.value_W_cols(L));
        b = theta(DuelingInfo.value_b_start(L):DuelingInfo.value_b_start(L)+DuelingInfo.value_b_len(L)-1);
        Z = W * V + b;
        if L < DuelingInfo.value_num_layers, V = max(0, Z); else, V = Z; end
    end
    Adv = shared_output;
    for L = 1:DuelingInfo.advantage_num_layers
        W = reshape(theta(DuelingInfo.advantage_W_start(L):DuelingInfo.advantage_W_start(L)+DuelingInfo.advantage_W_len(L)-1), DuelingInfo.advantage_W_rows(L), DuelingInfo.advantage_W_cols(L));
        b = theta(DuelingInfo.advantage_b_start(L):DuelingInfo.advantage_b_start(L)+DuelingInfo.advantage_b_len(L)-1);
        Z = W * Adv + b;
        if L < DuelingInfo.advantage_num_layers, Adv = max(0, Z); else, Adv = Z; end
    end
    Q = V + (Adv - mean(Adv));
end

function Q = Dueling_forward_vec_batch(theta, DuelingInfo, X)
    A = X;
    for L = 1:DuelingInfo.shared_num_layers
        W = reshape(theta(DuelingInfo.shared_W_start(L):DuelingInfo.shared_W_start(L)+DuelingInfo.shared_W_len(L)-1), DuelingInfo.shared_W_rows(L), DuelingInfo.shared_W_cols(L));
        b = theta(DuelingInfo.shared_b_start(L):DuelingInfo.shared_b_start(L)+DuelingInfo.shared_b_len(L)-1);
        A = max(0, W * A + b);
    end
    shared_output = A; V = shared_output;
    for L = 1:DuelingInfo.value_num_layers
        W = reshape(theta(DuelingInfo.value_W_start(L):DuelingInfo.value_W_start(L)+DuelingInfo.value_W_len(L)-1), DuelingInfo.value_W_rows(L), DuelingInfo.value_W_cols(L));
        b = theta(DuelingInfo.value_b_start(L):DuelingInfo.value_b_start(L)+DuelingInfo.value_b_len(L)-1);
        Z = W * V + b;
        if L < DuelingInfo.value_num_layers, V = max(0, Z); else, V = Z; end
    end
    Adv = shared_output;
    for L = 1:DuelingInfo.advantage_num_layers
        W = reshape(theta(DuelingInfo.advantage_W_start(L):DuelingInfo.advantage_W_start(L)+DuelingInfo.advantage_W_len(L)-1), DuelingInfo.advantage_W_rows(L), DuelingInfo.advantage_W_cols(L));
        b = theta(DuelingInfo.advantage_b_start(L):DuelingInfo.advantage_b_start(L)+DuelingInfo.advantage_b_len(L)-1);
        Z = W * Adv + b;
        if L < DuelingInfo.advantage_num_layers, Adv = max(0, Z); else, Adv = Z; end
    end
    Adv_mean = mean(Adv, 1);
    Q = V + (Adv - Adv_mean);
end