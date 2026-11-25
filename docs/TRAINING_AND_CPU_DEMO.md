# Training and reproduction evidence

The historical project used multilingual E5-small (118M parameters), not a generative LLM. See the [technical report](../TECHNICAL_REPORT.md) and the source scripts for the original training design.

| Setting | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| Training scope | Reference EIDs divisible by 10 | Same reference subset |
| Epochs | 1 | 1 |
| Learning rate | 5e-5 | 5e-5 |
| Batch size | 256 | 256 total |
| Token budget | 64 per record | 128 per pair |
| Objective | Multiple-negatives ranking | Binary classification |
| Negatives | Hard and in-batch | 10 hardest per reference |
| Seed | 20260926 | 20260926 |

Sources: [dense blocker](../src/step1_dense_blocker.py), [cross encoder](../src/step2_cross_encoder.py), [shared settings](../src/ber/settings.py). These are code-derived settings, not newly recovered training logs. Historical scores remain project-reported.

## Small CPU demonstration

From a Python 3.12 virtual environment at repository root:

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s research_tests -v
python examples/synthetic_matcher.py --output results/synthetic_cpu.json
```

The demonstration creates artificial names, addresses, near-copies and singleton groups. It trains a small LightGBM classifier on 200 groups, selects a decision threshold on 60 separate groups, and evaluates on 80 unseen groups. A fixed similarity/address rule is the baseline. This is a deliberately simplified demonstration of supervised matching and validation, not the full 91-feature matcher, encoder training or competition performance. No score is claimed until execution. CI saves the resulting JSON as an artifact. Production metric and normalization are separately exercised by regression tests.

## Original evidence still needed

Recover the authentic run reports, hardware and dependency versions, authorized leaderboard link or screenshot, and personally attributable commit/notebook evidence. Do not distribute private records or checkpoints without permission.
