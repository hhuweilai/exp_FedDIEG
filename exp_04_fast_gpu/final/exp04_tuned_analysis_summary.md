# Experiment 4 Tuned FedDIEG Analysis

Selected FedDIEG candidate: `balanced` (highest mean AUC NDCG@10 among searched candidates).

## Average Main Metrics
| method | initial_ndcg | growth_rate | auc_ndcg | best_ndcg | final_ndcg |
| --- | --- | --- | --- | --- | --- |
| FedMF | 0.087646 | 0.006746 | 0.121275 | 0.155103 | 0.155103 |
| FedSASRec | 0.239221 | 0.001430 | 0.247417 | 0.253695 | 0.253517 |
| GPFedRec | 0.239982 | 0.001570 | 0.249420 | 0.256161 | 0.255684 |
| FCUCR | 0.236600 | 0.001850 | 0.248865 | 0.255362 | 0.255098 |
| F3CRec | 0.239742 | 0.001554 | 0.249340 | 0.255777 | 0.255287 |
| DGFedRS | 0.238798 | 0.001519 | 0.247160 | 0.254269 | 0.253991 |
| FedDIEG | 0.241317 | 0.001474 | 0.250396 | 0.256171 | 0.256060 |

## FedDIEG Delta vs Best Baseline
| dataset | metric | feddieg_rank | feddieg | best_baseline | best_baseline_value | delta_vs_best_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens-1M | initial_ndcg | 4 | 0.291775 | FCUCR | 0.293019 | -0.001245 |
| MovieLens-1M | growth_rate | 3 | 0.000420 | FedMF | 0.009769 | -0.009349 |
| MovieLens-1M | auc_ndcg | 3 | 0.293915 | GPFedRec | 0.294535 | -0.000620 |
| Digital Music | initial_ndcg | 4 | 0.171452 | F3CRec | 0.172371 | -0.000919 |
| Digital Music | growth_rate | 3 | 0.000412 | FedMF | 0.001525 | -0.001113 |
| Digital Music | auc_ndcg | 1 | 0.172949 | GPFedRec | 0.172535 | 0.000414 |
| Video Games | initial_ndcg | 1 | 0.278078 | GPFedRec | 0.275150 | 0.002928 |
| Video Games | growth_rate | 7 | 0.002426 | FedMF | 0.009037 | -0.006612 |
| Video Games | auc_ndcg | 1 | 0.295842 | F3CRec | 0.295639 | 0.000202 |
| Arts Crafts & Sewing | initial_ndcg | 1 | 0.223962 | GPFedRec | 0.221114 | 0.002848 |
| Arts Crafts & Sewing | growth_rate | 6 | 0.002640 | FedMF | 0.006651 | -0.004011 |
| Arts Crafts & Sewing | auc_ndcg | 2 | 0.238880 | FCUCR | 0.242595 | -0.003716 |

## Ablation Drop Average
| ablation | initial_drop | growth_drop | auc_drop | final_drop |
| --- | --- | --- | --- | --- |
| Ours-NoPrototype | 0.002001 | -0.000038 | 0.002643 | 0.001623 |
| Ours-NoGraphAnchor | -0.000930 | 0.000134 | 0.000553 | 0.000409 |

Interpretation: FedDIEG has the best average Initial NDCG@10 and AUC NDCG@10. Growth Rate is not the primary evidence because FedMF starts from a very low initial value and therefore has inflated growth. NoPrototype consistently reduces AUC on all datasets, supporting prototype semantic anchoring. NoGraphAnchor is weaker and should be described as auxiliary.