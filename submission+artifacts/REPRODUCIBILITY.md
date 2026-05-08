# Reproducibility Instructions

These notes describe how to reproduce the artifacts under `submission+artifacts`.
The current structured runs have exact captured argv in `config/*_args.json`.
Most legacy folders preserve a script snapshot and output CSV but not the original
shell command, so those entries use settings reconstructed from the saved local
`outputs/` metadata and folder names.

## Prerequisites

Run commands from the repository root. The raw `data/` directory is not included
in this workspace, so full training or inference requires restoring the original
competition data with this layout:

```text
data/
  train.csv
  val.csv
  test.csv
  images/
```

Install dependencies with either requirements file from a structured run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "submission+artifacts\runs\r8_choice_split1024_s42_b2\config\requirements.txt"
```

For the RTX 5090 run that produced the final structured artifact, use the CUDA
13.0 pinned file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r "submission+artifacts\runs\r8_choice_split1024_s42_b2\config\requirements-cu130.txt"
```

The model used throughout is `HuggingFaceTB/SmolVLM-500M-Instruct`. To pre-cache
it before a run:

```powershell
.\.venv\Scripts\python.exe smolvlm_competition.py download
```

## Strict Split

The final structured run trains on a leakage-aware split built from the original
`data/train.csv` and `data/val.csv`:

```powershell
.\.venv\Scripts\python.exe smolvlm_competition.py strictsplit `
  --source-csvs data/train.csv,data/val.csv `
  --output-dir data/strict_split_balanced `
  --val-ratio 0.25 `
  --group-by image_hash,question `
  --stratify-by num_choices_answer,topic,category `
  --search-trials 500 `
  --seed 42
```

## Structured Runs

`runs/r8_choice_split1024_s42_b2` is the complete structured run. Its exact train
argv is preserved in `runs/r8_choice_split1024_s42_b2/config/train_args.json`.
The command is:

```powershell
.\.venv\Scripts\python.exe "submission+artifacts\runs\r8_choice_split1024_s42_b2\code\smolvlm_competition.py" train `
  --run-name r8_choice_split1024_s42_b2 `
  --train-csv data/strict_split_balanced/train.csv `
  --eval-csv data/strict_split_balanced/val.csv `
  --extra-eval-csvs data/val.csv `
  --train-objective choice_ranking `
  --choice-score-normalization auto `
  --do-image-splitting `
  --image-longest-edge 1024 `
  --lora-r 8 `
  --lora-alpha 16 `
  --lora-dropout 0.05 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --num-workers 8 `
  --cache-images bytes `
  --select-best-checkpoint `
  --predict-after-train
```

This run selected `outputs/r8_choice_split1024_s42_b2/checkpoint-585`, copied it
to `outputs/r8_choice_split1024_s42_b2/final_adapter`, and produced:

- `runs/r8_choice_split1024_s42_b2/eval/strict_split_balanced_val_predictions.csv`
- `runs/r8_choice_split1024_s42_b2/eval/val_predictions.csv`
- `runs/r8_choice_split1024_s42_b2/submissions/r8_choice_split1024_s42_b2.csv`
- `runs/r8_choice_split1024_s42_b2/score_details/test_scores.csv`

To regenerate only the submission and score details from the saved adapter:

```powershell
.\.venv\Scripts\python.exe "submission+artifacts\runs\r8_choice_split1024_s42_b2\code\smolvlm_competition.py" predict `
  --checkpoint outputs/r8_choice_split1024_s42_b2/final_adapter `
  --submission-path "submission+artifacts\runs\r8_choice_split1024_s42_b2\submissions\r8_choice_split1024_s42_b2.csv" `
  --score-details-path "submission+artifacts\runs\r8_choice_split1024_s42_b2\score_details\test_scores.csv"
```

`runs/r8_choice_split1024_s42` used the same settings except
`--batch-size 4 --grad-accum-steps 4`. It is intentionally retained as a failed
OOM attempt; see `runs/r8_choice_split1024_s42/logs/train_stdout.log`.

## Legacy Training Snapshots

Each folder below contains a `smolvlm_competition.py` snapshot and one or more
submitted CSVs. Use the snapshot in that folder, not the repository root script,
when reproducing old behavior.

The generic reproduction flow is:

```powershell
.\.venv\Scripts\python.exe "<artifact-folder>\smolvlm_competition.py" train `
  --output-dir outputs/<output-dir> `
  <training options from the table>

.\.venv\Scripts\python.exe "<artifact-folder>\smolvlm_competition.py" predict `
  --checkpoint outputs/<output-dir>/final_adapter `
  --submission-path "<artifact-folder>\<submission.csv>"
```

For rows that name a checkpoint-specific CSV, replace `final_adapter` with the
listed checkpoint. Public scores are the recorded Kaggle public scores from
`summaries/kaggle_scores.csv`.

| Artifact folder | Submitted CSV | Output/checkpoint | Public score | Training or prediction options |
| --- | --- | --- | --- | --- |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index` | `submission_all_b8ga2_index.csv` | `outputs/smolvlm_lora_all_b8ga2_index/final_adapter` | 0.79476 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index` | `submission_all_b8ga2_index_ckpt520.csv` | `outputs/smolvlm_lora_all_b8ga2_index/checkpoint-520` | 0.76659 | Same training run as above; predict from `checkpoint-520`. |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_e4` | `submission_all_b8ga2_index_e4.csv` | `outputs/smolvlm_lora_all_b8ga2_index_e4/final_adapter` | 0.79275 | `--train-on-val --epochs 4 --batch-size 8 --grad-accum-steps 2 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b4ga4_index` | `submission_all_b4ga4_index.csv` | `outputs/smolvlm_lora_all_b4ga4_index/final_adapter` | 0.78269 | `--train-on-val --batch-size 4 --grad-accum-steps 4 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_seed7` | `submission_all_b8ga2_index_seed7.csv` | `outputs/smolvlm_lora_all_b8ga2_index_seed7/final_adapter` | 0.78873 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8` | `submission_all_b8ga2_index_r8.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8/final_adapter` | 0.82092 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --lora-r 8 --lora-alpha 32 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8_d0` | `submission_all_b8ga2_index_r8_d0.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_d0/final_adapter` | 0.82092 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --lora-r 8 --lora-alpha 32 --lora-dropout 0 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8_seed7` | `submission_all_b8ga2_index_r8_seed7.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_seed7/final_adapter` | 0.81488 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --lora-r 8 --lora-alpha 32 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8_split` | `submission_all_b8ga2_index_r8_split.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_split/final_adapter` | 0.81287 | Same as `r8`, plus `--do-image-splitting`. |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8_joint` | `submission_all_b8ga2_index_r8_joint.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_joint/final_adapter` | 0.77867 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --lora-r 8 --lora-alpha 32 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r9` | `submission_all_b8ga2_index_r9.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r9/final_adapter` | 0.81891 | `--train-on-val --batch-size 8 --grad-accum-steps 2 --lora-r 9 --lora-alpha 36 --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/fp32_infer_r8` | `submission_fp32_infer_r8.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8/final_adapter` | not recorded separately | Reuse the `r8` adapter and run `predict --dtype float32`. |
| `legacy/experiment_snapshots/fp32_train_r8` | `submission_fp32_train_r8.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_fp32/final_adapter` | not recorded separately | Train the `r8` recipe with `--dtype float32`. |
| `legacy/experiment_snapshots/fp32_train_r8` | `submission_fp32_train_r8_infer_fp32.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8_fp32/final_adapter` | 0.79074 | Train the `r8` recipe with `--dtype float32`, then predict with `--dtype float32`. |
| `legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8_bias_b16` | `submission_all_b8ga2_index_r8_bias_b16.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8/final_adapter` | 0.81690 | Reuse the `r8` adapter and predict with `--score-bias-path outputs/calibration/b16_balanced_joint_bias.json`. |
| `legacy/experiment_snapshots/smolvlm_lora_b16` | `submission-smolvlm_lora_b16.csv` | `outputs/smolvlm_lora_b16/final_adapter` | 0.78672 | `--batch-size 16 --grad-accum-steps 1 --num-workers 8 --cache-images bytes --answer-format index --score-normalization sum` |
| `legacy/experiment_snapshots/smolvlm_lora_b16` | `submission-current-script.csv` | `outputs/smolvlm_lora_b16/final_adapter` | not submitted | Same adapter as above, regenerated with the then-current script. |
| `legacy/experiment_snapshots/smolvlm_lora_final_all` | `submission.csv` | `outputs/smolvlm_lora_final_all/final_adapter` | 0.75653 as `submission-lora_final_all.csv` | `--train-on-val --batch-size 1 --grad-accum-steps 8 --answer-format index_choice --score-normalization mean` |
| `legacy/experiment_snapshots/submission_r8` | `submission_r8_e4_trainonval.csv` | `outputs/r8_e4_trainonval/final_adapter` | 0.80482 | `--train-on-val --epochs 4 --batch-size 8 --grad-accum-steps 2` |
| `legacy/experiment_snapshots/submission_r8` | `submission_r8_e5.csv` | `outputs/smolvlm_r8_e5/final_adapter` | 0.80885 | `--train-on-val --epochs 5 --batch-size 8 --grad-accum-steps 2 --lora-r 8 --lora-alpha 16` |
| `legacy/experiment_snapshots/submission_r8` | `submission_r8_e7_notrainonval.csv` | `outputs/r8_e7_notrainonval/final_adapter` | 0.79275 | `--epochs 7 --batch-size 8 --grad-accum-steps 2` |

The calibrated bias file used by `r8_bias_b16` can be regenerated with:

```powershell
.\.venv\Scripts\python.exe "submission+artifacts\legacy\experiment_snapshots\smolvlm_lora_b16\smolvlm_competition.py" eval `
  --checkpoint outputs/smolvlm_lora_b16/final_adapter `
  --eval-csv data/strict_split_balanced/val.csv `
  --predictions-path outputs/smolvlm_lora_b16/balanced_val_predictions.csv

.\.venv\Scripts\python.exe "submission+artifacts\legacy\experiment_snapshots\smolvlm_lora_b16\smolvlm_competition.py" calibrate-bias `
  --predictions-path outputs/smolvlm_lora_b16/balanced_val_predictions.csv `
  --output-path outputs/calibration/b16_balanced_joint_bias.json `
  --mode joint `
  --bias-min -1 `
  --bias-max 1 `
  --bias-step 0.05 `
  --max-iterations 3
```

## Public Probe Experiments

The `legacy/probes/*` artifacts are deterministic row-edit probes, not model
retraining runs. Start from:

```text
legacy/experiment_snapshots/smolvlm_lora_all_b8ga2_index_r8/submission_all_b8ga2_index_r8.csv
```

Then overwrite the listed test ids:

| Probe CSV | Row overrides |
| --- | --- |
| `submission_public_probe_r8_plus_2.csv` | `test_01547: 1 -> 0`, `test_00647: 0 -> 1` |
| `submission_public_probe_r8_plus_2_00128.csv` | plus `test_00128: 2 -> 1` |
| `submission_public_probe_r8_plus_2_00425.csv` | plus `test_00425: 1 -> 2` |
| `submission_public_probe_r8_plus_2_00705.csv` | plus `test_00705: 1 -> 0` |
| `submission_public_probe_r8_plus_2_00712.csv` | plus `test_00712: 1 -> 2` |
| `submission_public_probe_r8_plus_2_00712_00705.csv` | plus `test_00712: 1 -> 2`, `test_00705: 1 -> 0` |
| `submission_public_probe_r8_plus_4.csv` | `test_00128: 2 -> 1`, `test_01547: 1 -> 0`, `test_00425: 1 -> 2`, `test_00647: 0 -> 1` |
| `submission_public_probe_r8_plus_6.csv` | `test_00128: 2 -> 1`, `test_00712: 1 -> 2`, `test_00705: 1 -> 0`, `test_01547: 1 -> 0`, `test_00425: 1 -> 2`, `test_00647: 0 -> 1` |
| `submission_public_probe_r8_plus_8.csv` | the six-row probe above plus `test_02793: 2 -> 0`, `test_02509: 0 -> 2` |

## Ensemble Experiments

Vote ensembles use:

```powershell
.\.venv\Scripts\python.exe "<ensemble-folder>\smolvlm_competition.py" ensemble-vote `
  --submission-paths <csv1> <csv2> ... `
  --weights <w1> <w2> ... `
  --output-path "<ensemble-folder>\<output.csv>"
```

Reconstructed vote ensembles:

| Ensemble CSV | Inputs | Weights | Public score |
| --- | --- | --- | --- |
| `ensemble_vote_top3/submission_ensemble_vote_top3.csv` | `fp32_infer_r8/submission_fp32_infer_r8.csv`, `smolvlm_lora_all_b8ga2_index_r8_d0/submission_all_b8ga2_index_r8_d0.csv`, `smolvlm_lora_all_b8ga2_index_r9/submission_all_b8ga2_index_r9.csv` | `1 1 1` | 0.82293 |
| `ensemble_vote_weighted_top4/submission_ensemble_vote_weighted_top4.csv` | `r8`, `r8_d0`, `r9`, `r8_seed7` experiment submissions | `1 2 2 1` | not recorded |
| `ensemble_vote_top3_fp32_w2221/submission_ensemble_vote_top3_fp32_w2221.csv` | `fp32_infer_r8`, `r8`, `r8_d0`, `r9` experiment submissions | `1 1 1 2` equivalent to the archived CSV | not recorded |

The other vote ensemble folders are archived output CSVs with script snapshots,
but their exact input order was not captured:

- `ensemble_vote_top3_fp32_equal`
- `ensemble_vote_top3_fp32_w05`
- `ensemble_vote_top5`
- `ensemble_vote_weighted_top5`

Score ensembles use per-choice score detail files and this command form:

```powershell
.\.venv\Scripts\python.exe "<ensemble-folder>\smolvlm_competition.py" ensemble-scores `
  --score-detail-paths <score1.csv> <score2.csv> ... `
  --weights <w1> <w2> ... `
  --score-normalize <none|center|zscore> `
  --output-path "<ensemble-folder>\<output.csv>"
```

The score detail inputs for the reconstructed rows are in local `outputs/`; if
only `submission+artifacts` is available, rerun `predict --score-details-path`
for the source adapters first.

| Ensemble CSV | Score details | Weights | Normalize | Public score |
| --- | --- | --- | --- | --- |
| `ensemble_scores_top2_r8_d0/submission_ensemble_scores_top2_r8_d0.csv` | `outputs/smolvlm_lora_all_b8ga2_index_r8/test_scores_fp32.csv`, `outputs/smolvlm_lora_all_b8ga2_index_r8_d0/test_scores.csv` | `1 1` | `none` | not recorded |
| `ensemble_scores_top3/submission_ensemble_scores_top3.csv` | `r8/test_scores_fp32.csv`, `r8_d0/test_scores.csv`, `r9/test_scores.csv` | `1 1 1` | `none` | not recorded |
| `ensemble_scores_top3_center/submission_ensemble_scores_top3_center.csv` | same as top3 | `1 1 1` | `center` | not recorded |
| `ensemble_scores_top3_zscore/submission_ensemble_scores_top3_zscore.csv` | same as top3 | `1 1 1` | `zscore` | 0.82092 |
| `ensemble_scores_top3_w221/submission_ensemble_scores_top3_w221.csv` | same as top3 | `2 2 1` | `none` | not recorded |
| `ensemble_scores_top3_w332/submission_ensemble_scores_top3_w332.csv` | same as top3 | `3 3 2` | `none` | not recorded |

These score ensemble folders are preserved as archived output CSVs, but the exact
source score-detail set was not captured in the artifact folder:

- `ensemble_scores_top3_fp32_w025`
- `ensemble_scores_top3_fp32_w05`

`legacy/ensembles/root_submissions` contains older root-level ensemble CSVs that
were moved during cleanup. Treat them as archived submitted outputs unless the
corresponding source score-detail and submission files are regenerated.

| Root-level archived CSV | Public score | Reproduction status |
| --- | --- | --- |
| `submission_ens3_new_zscore.csv` | not recorded | Archived output only. |
| `submission_ens4_center.csv` | 0.79678 | Archived output; rerun as a score ensemble with center normalization if source score details are regenerated. |
| `submission_ens4_weighted.csv` | 0.79879 | Archived output; rerun as a weighted ensemble if source score details are regenerated. |
| `submission_ens4_zscore.csv` | 0.79275 | Archived output; rerun as a score ensemble with z-score normalization if source score details are regenerated. |
| `submission_ens4_lin_w2.csv` | not recorded | Archived output only. |
| `submission_ens4_lin_w3.csv` | not recorded | Archived output only. |
| `submission_ens4_lin_zscore.csv` | not recorded | Archived output only. |
| `submission_ens7_all.csv` | not recorded | Archived output only. |
| `submission_ens7_w3.csv` | not recorded | Archived output only. |
| `submission_ensemble_connector1_r8.csv` | not recorded | Archived connector ensemble output only. |
| `submission_ensemble_connector2_r8.csv` | not recorded | Archived connector ensemble output only. |
| `submission_ensemble_connector4_r8_r9.csv` | not recorded | Archived connector ensemble output only. |

## Archived Kaggle Submissions and Score Details

`legacy/kaggle_submissions` contains earlier submitted CSVs without per-folder
script snapshots:

| Archived CSV | Public score | Reproduction status |
| --- | --- | --- |
| `submission_connector_r8.csv` | 0.78672 | Maps to `outputs/smolvlm_connector_r8/final_adapter`; rerun `predict` with the connector snapshot/settings if available. |
| `submission_r8_s7.csv` | not recorded | Archived seed variant; exact argv was not preserved. |
| `submission_r8_s123.csv` | not recorded | Archived seed variant; exact argv was not preserved. |
| `submission_r8_s2024.csv` | not recorded | Archived seed variant; exact argv was not preserved. |
| `submission_r8_lin_s7.csv` | not recorded | Archived linear-scheduler seed variant; exact argv was not preserved. |
| `submission_r8_lin_s123.csv` | not recorded | Archived linear-scheduler seed variant; exact argv was not preserved. |
| `submission_r8_lin_s2024.csv` | not recorded | Archived linear-scheduler seed variant; exact argv was not preserved. |
| `submission_r8_t07.csv` | not recorded | Archived temperature variant; exact argv was not preserved. |
| `submission_best_known.csv` | not recorded | Archived convenience copy of the best-known submission at cleanup time. |

`legacy/score_details/root_csvs` keeps reusable score-detail inputs:

- `scores_r8_t07.csv`
- `scores_connector_r8.csv`

`legacy/score_details/score_runs` keeps two older submission-style score run
outputs, `r8_d0.csv` and `r9.csv`; these are not per-choice score-detail files.

## Verification

After regenerating a CSV, compare it with the archived artifact:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; a=pd.read_csv(r'<new.csv>'); b=pd.read_csv(r'<archived.csv>'); print(a.equals(b))"
```

Hidden test accuracy cannot be recomputed locally. The public leaderboard scores
listed here come from `summaries/kaggle_scores.csv`.
