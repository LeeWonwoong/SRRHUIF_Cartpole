#!/bin/bash

# 파이썬 코드 파일명
SCRIPT="srrhuif_nd_vs_adam_debug_Qlandscape.py"

# 20가지 황금비율 & 한계 테스트 파라미터 셋 (r_std, P_max, Alpha)
PARAMS=(
    "1.5 0.014 0.99"
    "1.5 0.018 0.99"
    "1.8 0.013 0.99"
    "1.8 0.025 0.80"
    "1.8 0.025 0.99"
    "2.0 0.020 0.80"
    "2.0 0.020 0.99"
    "2.5 0.031 0.65"
    "2.5 0.038 0.80"
    "3.0 0.050 0.55"
    "3.0 0.050 0.70"
    "3.0 0.060 0.65"
    "3.5 0.061 0.48"
    "3.5 0.061 0.63"
    "4.0 0.080 0.42"
    "4.0 0.080 0.55"
    "4.5 0.101 0.36"
    "4.5 0.101 0.48"
    "5.0 0.125 0.33"
    "5.0 0.125 0.45"
)

BETAS=(2.0 3.0)

echo "========================================================="
echo "🚀 40-Combo Q-Landscape Experiments (Real-time Logs) Started..."
echo "========================================================="

count=1
total=$(( ${#BETAS[@]} * ${#PARAMS[@]} ))

for beta in "${BETAS[@]}"; do
    for p_set in "${PARAMS[@]}"; do
        
        read -r r_std p_max alpha <<< "$p_set"
        
        echo "---------------------------------------------------------"
        echo "[Run $count/$total] Beta=$beta | R=$r_std | P_max=$p_max | Alpha=$alpha"
        echo "---------------------------------------------------------"
        
        # ★ -u 옵션을 추가하여 출력을 버퍼링 없이 즉시 파일에 씁니다.
        python -u $SCRIPT --alpha "$alpha" --beta "$beta" --r_std "$r_std" --p_max "$p_max"
        
        ((count++))
    done
done

echo "========================================================="
echo "✅ All 40 Experiments Completed!"
echo "Check the 'results_cartpole' directory for the Q-Landscape plots."
echo "========================================================="
