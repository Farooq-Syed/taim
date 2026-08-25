# Real-data evaluation of TAIM on CICDDoS2019

**Date:** 2026-08-23 (revision — fold isolation, threshold calibration, day tagging, and a
substantially larger dataset). Tests the narrow question: **does TAIM's adaptive, time-aware
thresholding remain useful under family and capture-day shift on a public DDoS benchmark, versus
simple baselines?**

## Data

Official CICDDoS2019 flow captures (CC-BY-4.0, UNB). `scripts/build_real_windows.py` streams
a bounded per-family sample of the `01-12` (2018-12-01) and `03-11` (2019-03-11) captures,
adapts them via `src/cicddos_adapter.py` into per-(source-IP, **1-minute**) signal windows,
and tags each window with its `family` and the **capture identifier** as `day` (the capture
the flow belongs to, NOT a re-derived drifted timestamp). Output:
`data/cicddos_real_windows.csv` — **10,470 real windows, 2 capture days, 17 families**
(9,195 benign, 1,275 attack windows).

> The 1-minute bucket is intentional: each CICDDoS2019 family attack burst lasts only a few
> minutes, so coarser buckets (5+ min) collapse a family's attacks into 1–2 windows and give
> no per-family test support. At 1 minute, every family has **≥11 attack windows**
> (Syn 669, DrDoS_NTP 172, Portmap 79, … , WebDDoS 11). The per-device window count is still
> short (~3.2 windows/source-IP), which is a documented limitation for TAIM's temporal
> baseline.

## Protocol

`src/real_cicddos_eval.py` runs **strict splits** (never random rows):

- **family** — hold out an entire attack family; test = held-out family windows + a 20%
  benign split (benign held out from training).
- **day** — hold out an entire capture day (train on the other day, test on the held-out day).

**Fold isolation + calibration (reviewer-corrected).**

- **TAIM is fold-isolated** via `FastTaimDetector.run_fold(train, test)`: the adaptive
  baseline is warmed on the training rows only (updates on), then the test rows are scored
  against that frozen baseline (updates off). Held-out family/day telemetry never updates the
  baseline that scores it. Device identity is held constant across both phases.
- Only IsolationForest's `contamination` is tuned on an inner validation split. The
  RandomForest decision threshold is fixed at 0.5.
- The RF and TAIM **recall@FPR cutoffs are selected on a genuine validation split** — never
  on the test fold. RF uses a fresh model fit on a fit-split and scored on a validation-split;
  TAIM (stateful, cannot be re-fit per fold) is calibrated by a **chronological
  train/validation split within the training timeline**: the baseline is warmed on the earlier
  fraction and the cutoff is picked on the later fraction scored against that frozen baseline
  (a genuine out-of-warm-up operating point, not an in-sample one).
- Comparators all see the same 5 windowed signals (`bandwidth_mbps`, `conn_rate_ps`,
  `port_div`, `pkt_size_mean`, `app_req_ps`; bandwidth/app_req log1p-scaled).

## Results — strict family holdout (17 held-out families, every family ≥11 attack windows)

| comparator | F1 (±95%CI) | precision | recall | PR-AUC (±CI) | ROC-AUC (±CI) | recall@1%FPR | alerts |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **RandomForest (supervised)** | **0.792 (±0.049)** | 0.742 | 0.902 | **0.956 (±0.034)** | **0.992 (±0.010)** | 0.919 | 59.8 |
| IsolationForest (unsupervised) | 0.117 (±0.028) | 0.105 | 0.856 | — | — | — | 419.6 |
| fixed-rule baseline | 0.090 (±0.063) | 0.053 | 0.976 | — | — | — | 1,091.6 |
| TAIM (adaptive, fold-isolated) | 0.052 (±0.032) | 0.029 | 0.541 | 0.046 (±0.031) | 0.569 (±0.107) | 0.540 | 813.7 |

Per-family supervised RF: F1 0.52–0.90 and ROC-AUC 0.92–1.00 across all 17 families; TAIM F1
0.00–0.23 across all 17.

## Results — strict day holdout (2 capture days, correct tags)

| comparator | F1 | PR-AUC | ROC-AUC | recall@1%FPR |
|---|:--:|:--:|:--:|:--:|
| **RandomForest (supervised)** | **0.70** | — | 0.92 | — |
| TAIM (adaptive, fold-isolated) | 0.18 | — | ~0.55 | — |

(Per-day: RF F1 0.79/0.62, AUC 0.96/0.88; TAIM F1 0.17/0.20. Day-holdout n=2, so the day CI is
wide; the family holdout is the primary evidence.)

## Structural diagnostic (why TAIM is near-chance here)

TAIM is a **temporal** detector: it learns a per-device baseline over time and fires on
sustained deviations. On CICDDoS2019 the per-family captures are short bursts with only
**~3.2 windows per source-IP**, so there is essentially no per-device history for TAIM's
baseline (and no time-of-day regime). Even with fold isolation, TAIM's score separates
attacks poorly (ROC-AUC ≈ 0.54 on held-out families, and its recall@1%FPR cutoff hits an FPR
of 0.395 — i.e. the calibrated operating point does not preserve the validation FPR under
shift). IsolationForest and
RandomForest are snapshot models that do not need that history. This is a genuine property of
the benchmark + method, not a config artifact.

Additionally, TAIM's internal state is not re-run deterministic across separate process
invocations (a pre-existing property, unrelated to the fold-isolation fix), so exact-score
reprocessing should not be assumed; the reported aggregates use a fixed evaluation pass.

## Narrow claim

> On real CICDDoS2019 with strict family/day holdouts, **adaptive time-aware thresholding
> (TAIM) does not remain useful under distribution shift**: it is near-chance (ROC-AUC ≈ 0.54,
> PR-AUC ≈ 0.05) on unseen families, while a supervised baseline retains high performance
> on this bounded held-out-family subset
> (RF F1 0.79, ROC-AUC 0.99). TAIM's per-device temporal baseline has no purchase on short,
> family-isolated flows. This is a benchmark-transfer finding, not a deployment-level claim.

## Reproduce

```bash
python scripts/download_real_data.py --zips-bencorn          # data/real/csvs/*.zip
python scripts/build_real_windows.py --rows-per-family 1000000 --bucket-min 1 \
    --output data/cicddos_real_windows.csv
python src/real_cicddos_eval.py --input data/cicddos_real_windows.csv --split family \
    --metrics-output results/cicddos_family_eval.json
python src/real_cicddos_eval.py --input data/cicddos_real_windows.csv --split day \
    --metrics-output results/cicddos_day_eval.json
```

## Honest limits

- This is a bounded per-family sample, not the full 22 GB trace; splits are real but on a
  subset of each capture.
- Only 2 capture days → the day-holdout CI is wide (n=2); the family-holdout (n=17) is the
  primary evidence.
- TAIM's poor result is partly benchmark-structural (very short per-device history), so it
  should be read as "TAIM does not transfer to this benchmark," not "TAIM is useless." A
  longer-horizon real trace (e.g. LANL or a multi-day capture) is the correct next test.
