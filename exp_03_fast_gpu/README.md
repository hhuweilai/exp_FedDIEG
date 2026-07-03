# Experiment 3 Fast GPU

Experiment 3 evaluates FedDIEG mechanism ablations under chronological drift.

Variants:

- `Ours-Plain`: no temporal consistency, no structural edge consistency.
- `Ours-NoTemp`: structural edge consistency only.
- `Ours-NoStruct`: temporal consistency only.
- `Ours-Full`: temporal consistency and structural edge consistency.

Primary metrics:

- `max_drop`: peak-to-trough drop over evaluated NDCG@10 values.
- `boundary_max_drop`: maximum NDCG@10 drop at window transitions.
- `avg_ecr`: average edge change rate of the dynamic client graph.
- `loss_oscillation`: local training loss oscillation.
- `avg_window_ndcg`: average final NDCG@10 over windows, reported to keep stability claims tied to recommendation quality.

Use `run_exp03_fast_gpu.py` on the server. The script is GPU-batched and follows the aligned data, candidate, negative sampling, participation, and SASRec backbone protocol used by the fast Experiment 1/2 runners.
