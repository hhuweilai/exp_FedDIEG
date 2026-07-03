#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/exp_FedDIEG
/root/miniconda3/bin/python exp_01_fast_gpu/run_exp01_fast_gpu.py \
  --dataset "amazon-video games" \
  --methods FedSASRec GPFedRec FCUCR TarDGR FedDIEG \
  --rounds 600 \
  --eval-every 10 \
  --clients-per-round 5062 \
  --microbatch-size 512 \
  --eval-batch-size 2048 \
  --out-dir exp_01_fast_gpu/final_video_games_600r
