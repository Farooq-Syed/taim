# TAIM — Time-Aware Incident Mitigation

A router-resident network-anomaly detector that watches for **DDoS / man-in-the-middle**
attacks on a LAN and responds with a *graduated mitigation ladder* instead of instantly
cutting people off. It learns each device's normal traffic pattern *by time of day*, fuses
several signals to cut false alarms, and escalates responses slowly enough to give operators
time.

This repo is the full simulation-and-evaluation framework: a synthetic LAN traffic generator,
the detector pipeline, and an honest evaluation harness (regular-split **and** walk-forward
tests, plus a randomized robustness sweep) that checks for overfitting instead of only
reporting good numbers on one dataset.

> **Short version for humans:** I wanted to know whether a router could learn *how its own
> network behaves* and quietly step in — throttling, not disconnecting — before a flood
> becomes an outage. The answer so far: yes on a LAN, with honest caveats. The failures are
> documented as carefully as the successes.

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
  ml_experiment.py   A/B/C comparison: current vs windowed vs ML autoencoder
  temporal.py        windowed-mean + PCA-autoencoder temporal scorers
  real_data.py       real-data pipeline (NSL-KDD flows → signals → detector)
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

## Machine-learning experiment (documented negative result)

We tested whether adding an unsupervised ML model would beat the hand-crafted
detector. Three systems were compared on the same evaluation harness:

- **A** — the current detector (control)
- **B** — + windowed-mean temporal scorer (non-ML)
- **C** — + PCA autoencoder over z-score windows (the ML idea)

| test | A F1 | B F1 | C F1 | C FPR |
|---|---|---|---|---|
| standard sweep (30 random envs) | 0.864 | 0.858 | **0.082** | 52–69% |
| strict sweep (25 envs: flash crowds, weaker/noisier) | 0.437 | 0.433 | **0.050** | 52% |
| real data (NSL-KDD flows) | 0.29 | 0.29 | 0.29 | 1.9% |

**Result: the ML autoencoder was trashed.** Its temporal signal genuinely
improved lowslow *recall* (91% → 97%) and caught every volumetric/flood/syn
window — but its reconstruction-error threshold is catastrophically fragile to
normal traffic drift and legitimate flash crowds, producing a 52–69% false
positive rate and collapsing F1 to ~0.05. On real NSL-KDD data it added
nothing. The experiment code is kept in `src/temporal.py`, `src/ml_experiment.py`
and `src/real_data.py` as evidence.

The strict and real-data tests surfaced two real (non-ML) weaknesses that are
now the priority:

1. **Legitimate flash crowds get flagged** (FPR 2.5% in the strict sweep) — the
   broad-activity gate cannot yet distinguish an all-device 2–3× *legit* spike
   from a low-and-slow attack.
2. **Real attack signatures differ from the simulated ones** — NSL-KDD attacks
   show *smaller* packets, *lower* bandwidth and more diverse services, while
   the detector is tuned to bandwidth-flood DDoS (bandwidth-up + ≥2 signals).

## Current limitations and known failures

These are the things we have *tried and that do not fully work yet* — reported the same way
the successes are, because a research project that hides its failures teaches nothing:

- **Low-and-slow attacks are the hard case.** `lowslow` (a distributed attack where every
  device only raises traffic ~3×) is detected in ~82% of windows across random environments.
  When it lands on naturally quiet traffic it can fall below the statistical noise floor, and
  the baseline can gradually absorb it. This is a genuine, unsolved boundary.
- **Legitimate flash crowds look like attacks.** A sudden all-device spike (e.g., a product
  launch) triggers the broad-activity gate, producing ~2.5% false positives in the strict
  test. Volume alone cannot always separate "everyone is busy" from "everyone is attacked."
- **Real-world attack signatures differ from our simulations.** On real NSL-KDD flows, attack
  records show *smaller* packets, *lower* bandwidth and more diverse services — the opposite
  of our modelled bandwidth floods. The detector is currently tuned to volumetric DDoS.
- **The ML experiment failed.** Adding an autoencoder improved recall but collapsed precision
  (52–69% false positives). See the ML section below.
- **Synthetic-data-only validation.** No real multi-day per-device NetFlow trace has been run
  through the pipeline yet; the NSL-KDD exercise is a partial, imperfect substitute.

Each of these has a corresponding item in the future-plans list.

## Future plans

1. **Real-data validation** — ingest real network flow/SNMP data (e.g., CIC-IDS2017 or a
   company's NetFlow) and repeat the same regular/walk-forward evaluation. An end-to-end
   NSL-KDD pipeline already exists in `src/real_data.py` (download the CSV to `data/real/`
   and run it).
2. **Fix flash-crowd false positives** — teach the broad-activity gate to distinguish a
   legitimate all-device spike from a low-and-slow attack (e.g., cross-check duration vs
   ramp shape, or require protocol-mix change in addition to volume).
3. **Broader real-world signal set** — add signatures for real attack archetypes that are
   *not* bandwidth floods (service scans/probes: more distinct ports + smaller packets),
   so the detector generalizes beyond volumetric DDoS.
4. **Adaptive calibration** — replace static thresholds (noise floor, elevation levels) with
   per-network auto-calibration, so a quiet office and a noisy campus both work without
   manual tuning. This targets the remaining ~18% of lowslow misses.
5. **Time-windowed scoring** — average signals over a short rolling window before scoring to
   reduce per-step noise (the ML experiment showed this temporal signal is real, but it must
   be implemented with robust, drift-resistant thresholds).
6. **Feedback loop** — feed confirmed-attack windows back into the baseline model so a second
   attack of the same shape is caught faster (and today's attack isn't learned as "normal").
7. **Deployment study** — port to an embedded router (e.g., OpenWrt) and measure real latency,
   memory, and the effect of the graduated QoS caps / authenticated deauth on real clients.
8. **More attack classes** — botnet C2 chatter, DNS amplification, slowloris-style
   application-layer attacks, and encryption-blind traffic fingerprints (JA3/JA4).
9. **Revisit ML — only as a calibrated secondary channel** — the autoencoder's temporal
   signal helps recall, so if ML returns it must be a drift-resistant, secondary signal
   feeding the existing explainable ladder, never the primary decision maker.

## Academic integrity — use of AI assistance

This project was developed as part of doctoral research, so the use of AI is disclosed
honestly here (and should be included in any thesis per the relevant university policy):

- **Conceived and directed by the human.** The core ideas — behaviour-based detection instead
  of trusting spoofable device IDs, time-of-day baselines, multi-signal fusion, the graduated
  mitigation ladder, and the anti-overfitting evaluation methodology — were the author's. Every
  decision about what to build, test, keep, or reject (including rejecting the ML approach and
  the hardware-ID idea) was made and validated by the author.
- **AI assisted as a tool.** The `opencode` coding assistant helped translate the design into
  code, run and debug experiments, profile and vectorize the detector, and draft parts of this
  documentation. All of its output was reviewed, tested, and edited by the author.
- **Results are reported with their failures.** No numbers were tuned on test data — the
  configuration was frozen before the unseen-environment and robustness tests — and the known
  limitations and the negative ML result are documented alongside the positive results.

---

*Educational / research project for doctoral work. Simulations and synthetic data only — no
real networks or attacks were used or harmed. See "Academic integrity" above for the AI-use
disclosure.*
