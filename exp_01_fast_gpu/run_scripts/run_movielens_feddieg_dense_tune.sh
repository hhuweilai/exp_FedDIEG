#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/exp_FedDIEG

# Dense-data FedDIEG tuning for MovieLens-1M.
# Only reruns Ours; baselines remain from final_movielens_1m_600r.
# These settings weaken graph/short-term injection because MovieLens has long user histories.
/root/miniconda3/bin/python exp_01_fast_gpu/run_exp01_fast_gpu.py \
  --dataset "movielens-1m" \
  --methods FedDIEG \
  --rounds 600 \
  --eval-every 10 \
  --clients-per-round 604 \
  --microbatch-size 512 \
  --eval-batch-size 2048 \
  --feddieg-graph-mix 0.03 \
  --feddieg-neighbor-temp 0.05 \
  --feddieg-min-sim 0.20 \
  --feddieg-tail-power 4.0 \
  --feddieg-final-loss-weight 0.05 \
  --feddieg-memory-momentum 0.8 \
  --feddieg-short-score-mix 0.0 \
  --feddieg-neighbors 10 \
  --feddieg-warmup-rounds 100 \
  --out-dir exp_01_fast_gpu/final_movielens_1m_feddieg_dense_tune_600r
