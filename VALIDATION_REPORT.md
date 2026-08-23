# TAIM Reproduction and Validation Report

## Claim Boundary

TAIM currently demonstrates a causal detector and mitigation state machine on generated LAN
traffic. The evidence supports reproducibility across synthetic environments. It does not yet
support claims of operational effectiveness, MITM detection, or safety of automated blocking on
a production router.

The NSL-KDD adapter is a legacy public-benchmark exercise. It constructs a new timeline by
splicing labeled records and therefore is not an intact packet or flow-trace replay. Because the
official source no longer distributes NSL-KDD, that experiment was not rerun in this validation.

## Reproduction Commands

```text
python -m pytest tests -q
python src/run_evaluation.py
python src/final_validation.py
python src/robustness.py
python src/ml_experiment.py --envs 30 --seed 777
python src/ml_experiment.py --strict --envs 25 --seed 777
```

All executable paths are repository-relative. Generated data are excluded from Git; result CSVs
are retained so reviewers can inspect individual environments rather than only summary values.

## Independently Rerun Results

| Evaluation | TPR | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| Phase 5 regular | 0.9492 | 0.0076 | 0.8320 | 0.8867 |
| Phase 5 walk-forward | 0.9231 | 0.0100 | 0.8060 | 0.8606 |
| Phase 6 unseen regular | 0.9077 | 0.0032 | 0.8994 | 0.9035 |
| Phase 6 unseen walk-forward | 0.9313 | 0.0040 | 0.8917 | 0.9111 |

The 60-environment sweep reproduced mean F1 0.888, median F1 0.903, and a wide 0.372-0.974
range. Window detection was 79/82 volumetric, 78/78 flood, 70/70 SYN, and 62/76 low-and-slow.
The minimum is important: average performance conceals substantial environment sensitivity.

The paired 30-environment ML experiment reproduced mean F1 values of 0.864 for TAIM, 0.858 for
the windowed baseline, and 0.082 for PCA, whose mean FPR was 68.9%. In the 25-environment strict
experiment, the corresponding mean F1 values were 0.396, 0.392, and 0.061. TAIM's strict mean
FPR was 3.1%, and its low-and-slow window recall fell to 13/22 (59%). The per-environment
evidence is stored in `results/ml_experiment_standard.csv` and
`results/ml_experiment_strict.csv`.

## Failure Analysis

- Low-and-slow detection is the weakest attack class at 82% window recall.
- The worst Phase 5 and Phase 6 walk-forward folds reached 19.3% and 37.9% FPR even though
  aggregate FPR was much lower.
- The simulator defines both training conditions and attacks, so generator-detector coupling
  remains a major threat to external validity.
- The mitigation ladder has not been tested against router QoS, queueing, latency, collateral
  damage, or operator response in a controlled deployment.
- Historical 150-device and 500-device throughput numbers lack a retained benchmark harness and
  raw timing records, so they are excluded from the independently reproduced evidence.
- NSL-KDD lacks the per-device, timestamped LAN semantics assumed by TAIM; its constructed replay
  cannot close the external-validation gap.

## Research-Ready Next Experiment

Use timestamped CICDDoS2019 CSV flows as a frozen public benchmark. Define the feature mapping and
exclusions before scoring, split by capture day rather than random rows, publish source file
hashes, and report per-family recall, false alarms per hour, detection delay, worst temporal fold,
and abstentions caused by missing features. Follow this with an authorized OpenWrt testbed study
that measures mitigation latency and benign-flow harm. These are prerequisites for a defensible
operational-effectiveness claim, not optional polish.
