# Experiment 5 Fast GPU

Experiment 5 runs a one-factor-at-a-time sensitivity analysis for the four
FedDIEG hyperparameters requested by the advisor:

| Paper symbol | Code parameter | Default | Sweep values |
| --- | --- | ---: | --- |
| `K` | `feddieg_neighbors` | `10` | `5, 10, 20, 40` |
| `rho` | `edge_ema` | `0.80` | `0.0, 0.3, 0.5, 0.8, 0.9` |
| `lambda_1` | `lambda_temp` | `0.01` | `0.0, 0.005, 0.01, 0.05, 0.1` |
| `lambda_2` | `lambda_struct` | `0.20` | `0.0, 0.05, 0.1, 0.2, 0.4` |

The runner reuses `exp_03_fast_gpu/run_exp03_fast_gpu.py` with the
chronological-window protocol.  Only `Ours-Full` is trained for each setting.
The default configuration is trained once per dataset and reused when building
per-parameter tables, so each dataset has 16 unique jobs.

Report metrics:

- `avg_window_ndcg`
- `max_drop`
- `avg_ecr`
- `stability_score`

## Run

Full four-dataset run:

```bash
cd ~/autodl-tmp/exp_FedDIEG
bash exp_05_sensitivity/run_exp05_all.sh
```

Pilot run on two datasets:

```bash
cd ~/autodl-tmp/exp_FedDIEG
bash exp_05_sensitivity/run_exp05_pilot.sh
```

Rebuild tables and figures after interrupted or manually completed jobs:

```bash
cd ~/autodl-tmp/exp_FedDIEG
/root/miniconda3/bin/python exp_05_sensitivity/run_exp05_sensitivity.py --aggregate-only
```

Write commands without launching training:

```bash
cd ~/autodl-tmp/exp_FedDIEG
/root/miniconda3/bin/python exp_05_sensitivity/run_exp05_sensitivity.py
```

## Outputs

Main tables are written to:

```text
exp_05_sensitivity/final/
  exp05_raw_summary.csv
  exp05_sensitivity_summary.csv
  exp05_mean_by_value.csv
  exp05_sensitivity_rank.csv
  exp05_missing_jobs.csv
```

Figures and JSON values are written to:

```text
figures/exp05_sensitivity/
  fig8_exp05_avg_window_ndcg.png
  fig9_exp05_stability_score.png
  fig10_exp05_max_drop.png
  fig11_exp05_avg_ecr.png
  table7_exp05_sensitivity_values.json
```
