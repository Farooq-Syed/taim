# Real-data evaluation of TAIM on CICDDoS2019

**Date:** 2026-08-23. Tests the narrow question from the publication roadmap: **does
TAIM's adaptive, time-aware thresholding remain useful under realistic shift on a real
public DDoS benchmark, versus simple baselines?**

## Data

Official CICDDoS2019 flow captures (CC-BY-4.0, UNB). `scripts/build_real_windows.py`
streams a bounded per-family sample of the `01-12` (2018-12-01) and `03-11` (2019-03-11)
captures, adapts them via `src/cicddos_adapter.py` into per-(source-IP, 15-min) signal
windows, and tags each window with its `family` and capture `day`. Output:
`data/cicddos_real_windows.csv` — **4,471 real windows, 2 capture days, 17 families**
(4,367 benign, 104 attack windows).

> This is a real-trace windowed evaluation, not a constructed replay. The caveat is that
> the 15-class mirror is a feature-selected subset; for the full-trace run use the entire
> capture (see `scripts/download_real_data.py`). The per-device window count is short
> (~3.2 windows per source-IP), so TAIM's temporal history is thin on this benchmark.

## Protocol

`src/real_cicddos_eval.py` runs a **strict split** (never random rows):

- **family** — hold out an entire attack family; test = held-out family windows + a 20%
  benign split (benign held out from training).
- **day** — hold out an entire capture day (train on the other day, test on the held-out day).

Only IsolationForest's `contamination` is tuned on an inner validation split; the
RandomForest and TAIM operating thresholds are fixed (RF 0.5; the family split's
recall@1%FPR cutoff is selected on validation, never the test fold). Comparators all see
the same 5 windowed signals: `bandwidth_mbps`, `conn_rate_ps`, `port_div`, `pkt_size_mean`,
`app_req_ps` (bandwidth / app_req are log1p-scaled for numerical stability).

## Results — strict family holdout (17 held-out families)

| comparator | F1 | precision | recall | PR-AUC | ROC-AUC | recall @ 1%FPR | alerts |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **RandomForest (supervised)** | **0.772** | 0.703 | 0.911 | **0.967** | **0.999** | 0.890 | 5.3 |
| IsolationForest (unsupervised) | 0.421 | 0.317 | 0.776 | — | — | — | 8.9 |
| TAIM (adaptive) | 0.027 | 0.017 | 0.238 | 0.015 | 0.577 | 0.326 | 51.6 |
| fixed-rule baseline | 0.019 | 0.010 | 1.000 | — | — | — | 582 |

Per-family supervised RF: F1 0.47–0.89, ROC-AUC 0.99–1.00 across all 17 families; TAIM F1
0.00–0.11 across all 17.

## Results — strict day holdout (2 capture days)

| comparator | F1 | PR-AUC | ROC-AUC | recall @ 1%FPR | alerts |
|---|:--:|:--:|:--:|:--:|:--:|
| **RandomForest (supervised)** | **0.762** | 0.895 | 0.970 | 0.532 | 31.5 |
| IsolationForest (unsupervised) | 0.638 | — | — | — | 73.5 |
| TAIM (adaptive) | 0.058 | 0.031 | 0.535 | 0.347 | 146.5 |
| fixed-rule baseline | 0.071 | — | — | — | 1497.5 |

## Structural diagnostic (why TAIM is near-chance here)

TAIM is a **temporal** detector: it learns a per-device baseline over time and fires on
sustained deviations. On CICDDoS2019 the per-family captures are short bursts with only
**~3.2 windows per source-IP**, so there is essentially no per-device history for TAIM's
baseline (and no time-of-day regime) — its in-domain score already separates poorly (mean
score 0.081 on attack vs 0.099 on benign; flags 4/104 attack windows). IsolationForest and
RandomForest are *snapshot* models and don't need that history. This is a genuine property
of the benchmark + method, not a config artifact; it is reported honestly.

## Narrow claim

> On real CICDDoS2019 with strict family/day holdouts, **adaptive time-aware thresholding
> (TAIM) does not remain useful under distribution shift**: it is near-chance
> (ROC-AUC ≈ 0.58, PR-AUC ≈ 0.02) on unseen families, while a supervised baseline
> generalizes strongly (RF ROC-AUC 0.999, F1 0.77). TAIM's per-device temporal baseline
> has no purchase on short, family-isolated flows. This is not a deployment-level claim;
> it is a benchmark-transfer finding.

## Reproduce

```bash
# 1. Download the official captures (needs HF mirror; see scripts/download_real_data.py)
python scripts/download_real_data.py --zips-bencorn   # -> data/real/csvs/*.zip
# 2. Build the windowed dataset
python scripts/build_real_windows.py --rows-per-family 700000 --bucket-min 15 \
    --output data/cicddos_real_windows.csv
# 3. Strict family and day holdouts
python src/real_cicddos_eval.py --input data/cicddos_real_windows.csv --split family \
    --metrics-output results/cicddos_family_eval.json
python src/real_cicddos_eval.py --input data/cicddos_real_windows.csv --split day \
    --metrics-output results/cicddos_day_eval.json
```

## Honest limits

- The windowed dataset is a bounded per-family sample, not the full 22 GB trace; the
  family/day splits are real but on a subset.
- Only 2 capture days → the day-holdout CI is wide (n=2); the family-holdout (n=17) is the
  primary evidence.
- TAIM's poor result is partly benchmark-structural (very short per-device history), so it
  should be read as "TAIM does not transfer to this benchmark," not "TAIM is useless."
  A longer-horizon real trace (e.g. LANL or a multi-day capture) is the correct next test.
