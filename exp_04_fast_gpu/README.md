# Experiment 4 Fast GPU

This folder runs the new-client onboarding pre-validation for FedDIEG's
prototype semantic anchoring mechanism with GPU-batched real training.

Protocol:
- Hold out 25% users as unseen new clients.
- By default, prioritize low-history users and keep the same split for all
  methods (`--new-user-strategy low_history`).
- Pretrain for 50 federated rounds on old clients only.
- Inject new clients with `d_init=3` visible interactions.
- Each adaptation round reveals up to 2 additional local interactions per new
  client until that client's held history is exhausted.
- All methods use one local/adaptation update pass per communication round by
  default.
- Track NDCG@10 and HR@10 over 0..10 local adaptation rounds.
- Main comparison: `FedMF`, `FCUCR`, `F3CRec`, `DGFedRS`, `FedDIEG`.
- Legacy extra baselines (`FedSASRec`, `GPFedRec`, `TarDGR`) remain available
  only when explicitly passed through `--methods`.
- Ablation: `Ours-NoPrototype`, `Ours-NoGraphAnchor`, `FedDIEG`.
- Optional diagnostic ablation: add `--include-short-ablation` to also run
  `Ours-NoShortGate`.

Main outputs:
- `trajectory.csv`
- `summary.csv`
- `ablation_trajectory.csv`
- `ablation_summary.csv`
- `ablation_contribution.csv`
- `ndcg_trajectory.png`
- `ablation_ndcg_trajectory.png`

Smoke test example:

```bash
cd /root/autodl-tmp/exp_FedDIEG
python exp_04_fast_gpu/run_exp04_fast_gpu.py \
  --dataset movielens-1m \
  --methods FedMF FCUCR F3CRec DGFedRS FedDIEG \
  --pretrain-rounds 2 \
  --adapt-rounds 2 \
  --max-eval-users 256 \
  --clients-per-round 128 \
  --microbatch-size 256 \
  --out-dir exp_04_fast_gpu/smoke/movielens-1m
```

Unified all-model full run:

```bash
cd /root/autodl-tmp/exp_FedDIEG
bash exp_04_fast_gpu/run_exp04_all_models_uniform.sh
```

The unified script trains `FedMF`, `FedSASRec`, `GPFedRec`, `FCUCR`, `F3CRec`,
`DGFedRS`, and `FedDIEG` on all four datasets with the same protocol and writes
to `exp_04_fast_gpu/final_uniform_all_models` by default. To overwrite the
legacy `final` folder instead, run with `BASE_OUT=.../exp_04_fast_gpu/final`.

Append only the newly added baselines to existing final result folders:

```bash
cd /root/autodl-tmp/exp_FedDIEG
bash exp_04_fast_gpu/run_exp04_append_baselines.sh
```

Tune FedDIEG only and overwrite its rows in the existing final result folders:

```bash
cd /root/autodl-tmp/exp_FedDIEG
bash exp_04_fast_gpu/run_exp04_tune_feddieg_only.sh
```

Single dataset full run example:

```bash
cd /root/autodl-tmp/exp_FedDIEG
/root/miniconda3/bin/python exp_04_fast_gpu/run_exp04_fast_gpu.py \
  --dataset movielens-1m \
  --methods FedMF FCUCR F3CRec DGFedRS FedDIEG \
  --pretrain-rounds 50 \
  --adapt-rounds 10 \
  --new-client-ratio 0.25 \
  --new-user-strategy low_history \
  --d-init 3 \
  --new-interactions-per-round 2 \
  --adapt-lr 0.0002 \
  --out-dir exp_04_fast_gpu/final/movielens-1m
```
