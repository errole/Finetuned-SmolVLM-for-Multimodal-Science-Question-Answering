# Submission Artifacts

This folder is organized into stable buckets:

- `final/`: quick pointer to the chosen final structured run and submission CSV.
- `runs/`: current structured experiment runs with configs, logs, metrics, submissions, score details, and code snapshots.
- `summaries/`: generated CSV summaries across local run logs, trainer histories, artifact inventory, cleanup moves, and Kaggle scores.
- `legacy/`: older submissions and one-off experiment snapshots moved out of the root folder without deleting them.
- `reference/`: static reference files, including the Kaggle raw score notes and sample submission.
- `REPRODUCIBILITY.md`: commands and notes for reproducing each structured run,
  legacy experiment snapshot, public probe, and ensemble artifact.

Primary final artifact:

- Submission CSV: `runs/r8_choice_split1024_s42_b2/submissions/r8_choice_split1024_s42_b2.csv`
- Run bundle: `runs/r8_choice_split1024_s42_b2/`
- Public Kaggle score: `0.76257`
- Reproduction notes: `REPRODUCIBILITY.md`

Useful summary files:

- `summaries/structured_run_metrics_summary.csv`: metrics for structured runs under `runs/`.
- `summaries/local_training_metrics_summary.csv`: metrics recovered from `outputs/*/trainer_state.json` and `outputs/*/best_checkpoint.txt`.
- `summaries/kaggle_scores.csv`: normalized version of `reference/kaggle_scores_raw.txt`.
- `summaries/kaggle_submission_index.csv`: Kaggle scores joined to matching local submission files after cleanup.
- `summaries/cleanup_manifest.csv`: every file or directory moved or removed during cleanup.
- `summaries/artifact_inventory.csv`: post-cleanup inventory with file sizes and CSV line counts.
