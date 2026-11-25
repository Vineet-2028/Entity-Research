# Reproduction guide

## 1. Minimal offline checks

From the repository root, run `python -m unittest discover -s tests -v` and `python -m compileall -q src tests`. These use the standard library only and check restored package layout, command help, path defaults and a synthetic packaging fixture. They do not reproduce the ML results.

## 2. Full training

Use Python 3.12 on Linux, one or more supported CUDA GPUs (original guide recommends >=12 GB), >=32 GB CPU RAM and >=25 GB free disk. Install the appropriate CUDA PyTorch build where necessary; install the pinned requirements and run `pip check`. Record actual package versions, GPU, driver and seeds. Do not silently replace incompatible pins and call the run an exact reproduction.

Provide the authorized challenge TSVs: `train/train_source1.tsv`, `train/train_source2.tsv`, `train/train_source3.tsv`, `train/train_ground_truth.tsv`, and `test/test_source1.tsv`, `test/test_source2.tsv`, `test/test_source3.tsv`. Use explicit absolute `--data-dir`, `--work-dir`, `--out-dir` to avoid ambiguity. Follow [the run guide](../read_before_run.md) and [the technical report](../TECHNICAL_REPORT.md).

For a valid run, retain `dense_report.json`, `ce_train_report.json`, `ce_dev_report.json`, `report_dev.json`, and `report_finalize.json`, plus environment information and command lines. Keep raw data, checkpoints and identifying records private unless authorized. The historical leaderboard claim is not independently verified in this repository.

## 3. Remaining checks

Actual installation, GPU training, original artifact hashes, leaderboard confirmation and individual contribution attribution need original run evidence. The smoke test does not establish these. The competition-specific ZIP still needs the official validator and authorized real outputs before submission.
