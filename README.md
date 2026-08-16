# TAIM — Time-Aware Incident Mitigation

A router-resident network-anomaly detector that watches for **DDoS / man-in-the-middle** style
attacks on a LAN and responds with a *graduated mitigation ladder* instead of instantly cutting
people off. It learns each device's normal traffic pattern *by time of day*, fuses several
signals to cut false alarms, and escalates responses slowly enough to give operators time.

This repo is the full simulation-and-evaluation framework: a synthetic LAN traffic generator,
the detector pipeline, and an honest evaluation harness (regular-split **and** walk-forward
tests, plus a randomized robustness sweep) that checks for overfitting instead of only
reporting good numbers on one dataset.

---

## Why this project exists

A DDoS or a machine-in-the-middle attack is hard to stop *after* it starts. The original idea
was: instead of trusting a device's ID (MAC/hardware ID — which attackers can spoof), learn how
each device normally behaves and flag *behaviour*, not identity. And instead of dropping traffic
instantly (which also drops legitimate users), throttle it in stages: **watch → soft cap →
hard cap → drop**. That staged response is what buys you the "preparation time" a live network
needs.

## How it works

```
 signals (bandwidth, conn rate, protocol mix, packet size, app reqs)
   │  (per device, every 5-15 min)
   ▼
 Time-window baseline engine        learns "normal" per device + hour-of-day,
                                     outlier-resistant so attacks don't poison it
   ▼
 Multi-signal fusion                needs ≥2 signals elevated to fire → fewer
                                     false alarms (packet-size *drop* counts too)
   ▼
 Broad-activity gate                if many devices are elevated at once, it's a
                                     volumetric/low-and-slow attack on the whole LAN
   ▼
 Graduated response ladder          watch → soft cap → hard cap → drop, with
                                     graceful de-escalation when things recover
```

Everything runs **causally** (score-then-update), so there is no look-ahead — the detector
never "peeks" at future data, exactly like a live system.

## Repository layout

```
src/
  data_gen.py        synthetic LAN traffic + 4 attack types (flood, syn, volumetric, lowslow)
  baseline.py        time-of-day baseline engine (outlier-resistant EWMA)
  scoring.py         multi-signal fusion (≥2 signals to fire, direction-aware)
  ladder.py          graduated response state machine (incl. sustained-signal path)
  detector.py        reference detector (clear, readable)
  fast_detector.py   vectorized detector — 12× faster, functionally identical
  evaluate.py        regular-split vs walk-forward evaluation harness
  run_evaluation.py  Phase-5 results (tuning environment)
  final_validation.py Phase-6 results (unseen environment)
  robustness.py      60-environment randomized robustness sweep (anti-overfitting)
  plot_utils.py      report plots
tests/               pytest suite (46 tests)
results/             CSVs + plots from each evaluation
```

## How to run

Requires Python 3.12+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest`.

```bash
pip install -r requirements.txt

# unit + integration tests (46)
python -m pytest tests/ -q

# Phase 5: regular split vs walk-forward on the tuning environment
python src/run_evaluation.py

# Phase 6: same config, brand-new unseen environment (no re-tuning)
python src/final_validation.py

# Anti-overfitting sweep: 60 random environments
python src/robustness.py
```

The 42-day dataset is generated automatically if missing (`python src/data_gen.py`).

## Results

### Phase 5 — tuning environment (10 devices, 42 days)

Regular split vs walk-forward gave essentially the **same** result (F1 ≈ 0.86–0.89), which is
itself a good sign: no warm-up dependence, no leakage. All four attack types were detected and
escalated to stage 4 (drop), floods within ~45 min.

| regime | TPR | FPR | F1 |
|---|---|---|---|
| regular split | 0.949 | 0.008 | 0.887 |
| walk-forward | 0.923 | 0.010 | 0.861 |

### Phase 6 — unseen environment (15 devices, 56 days, different interval/noise/attacks)

Same untuned config. The unseen network performed **as well as or better than** the tuning
environment — the strongest evidence that the detector generalises.

| regime | TPR | FPR | F1 |
|---|---|---|---|
| regular split | 0.908 | 0.003 | 0.904 |
| walk-forward | 0.931 | 0.004 | 0.911 |

### Robustness sweep — 60 random environments (anti-overfitting check)

The config was developed on one dataset, so we stress-tested it on 60 random networks
(random size, interval, noise, diurnal shape, weekend behaviour, attack schedules) with **zero
per-environment tuning**:

| metric | mean | median | min | max |
|---|---|---|---|---|
| F1 | 0.888 | 0.903 | 0.372 | 0.974 |
| FPR | 0.004 | 0.003 | 0.000 | 0.014 |

Window detection: **flood 100%, syn 100%, volumetric 96%, lowslow 82%** of all attack windows.

This sweep is exactly what caught the overfitting: the first version had a long tail of bad
environments (FPR up to 8%, F1 as low as 0.11). A *sustained broad-activity gate* (requiring
several consecutive steps before escalating) removed the phantom attacks and brought the worst
FPR down to 1.4%.

### Scale — real-company size

| network | rows | runtime | throughput | F1 |
|---|---|---|---|---|
| 150 devices, 90 days, 5-min | 3.9 M | 21 s | 183 K rows/s | 0.956 |
| 500 devices, 1 year, 5-min | 52.6 M | 210 s | 250 K rows/s | 0.962 |

0.58 s per day of data means the detector runs ~250× faster than real time — practical for a
live edge router. The vectorized `FastTaimDetector` is verified byte-equivalent (identical
stages/flags) to the reference implementation.

## Honest limitations

- **Low-and-slow attacks are the hard case.** `lowslow` (a distributed attack where every
  device only raises traffic ~3×) is detected in ~82% of windows across random environments.
  It can fall below the statistical noise floor when it lands on naturally quiet traffic, and
  it's a known boundary case.
- **Brief targeted attacks** (a single device flooding) can be detected *late* in some
  environments — detection quality is good, but latency varies.
- The simulation is synthetic. Real per-device NetFlow/SNMP data would be the next validation
  step; the detector only needs per-device traffic metrics per interval.

## Future plans

1. **Real-data validation** — ingest real network flow/SNMP data (e.g., CIC-IDS2017 or a
   company's NetFlow) and repeat the same regular/walk-forward evaluation.
2. **Adaptive calibration** — replace static thresholds (noise floor, elevation levels) with
   per-network auto-calibration, so a quiet 20-device office and a noisy 500-device campus are
   both handled without manual tuning. This targets the remaining 18% lowslow misses.
3. **Time-windowed scoring** — average signals over a short rolling window before scoring to
   reduce per-step noise and catch sustained attacks earlier (improves lowslow latency).
4. **Feedback loop** — feed confirmed-attack windows back into the baseline model so a second
   attack of the same shape is caught faster (and today's attack isn't learned as "normal").
5. **Deployment study** — port to an embedded router (e.g., OpenWrt) and measure real latency,
   memory, and the effect of the graduated QoS caps / authenticated deauth on real clients.
6. **More attack classes** — botnet C2 chatter, DNS amplification, slowloris-style
   application-layer attacks, and encryption-blind traffic fingerprints (JA3/JA4).

---

*Educational / research project. Simulations and synthetic data only — no real networks or
attacks were used or harmed.*
