"""Parameter sensitivity: perturb each detector config knob slightly and
measure the impact on Phase-5 metrics. Shows which knobs are safe vs brittle."""
import sys
sys.path.insert(0, r'C:\Users\Farooq Syed\taim')
import pandas as pd
from src.data_gen import default_config, generate_dataset
from src.fast_detector import FastTaimDetector
from src.detector import DetectorConfig
from src.baseline import BaselineConfig
from src.scoring import FusionConfig
from src.ladder import LadderConfig

df, _ = generate_dataset(default_config())

def met(cfg):
    out = FastTaimDetector(cfg).run(df)
    y = out['is_attack'].astype(bool); p = out['flagged'].astype(bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    tpr = tp/(tp+fn); fpr = fp/(fp+int((~p & ~y).sum())); prec = tp/(tp+fp)
    f1 = 2*prec*tpr/(prec+tpr) if (prec+tpr) else 0
    return tpr, fpr, f1

base = DetectorConfig()
b_tpr, b_fpr, b_f1 = met(base)
print(f'baseline:              TPR={b_tpr:.3f} FPR={b_fpr:.4f} F1={b_f1:.3f}')
print('-' * 70)

def perturb(name, make_cfg):
    tpr, fpr, f1 = met(make_cfg())
    print(f'{name:28s} TPR={tpr:.3f} FPR={fpr:.4f} F1={f1:.3f}  (dF1={f1-b_f1:+.3f})')

# baseline knob
perturb('alpha=0.03', lambda: DetectorConfig(baseline=BaselineConfig(
    min_samples=3, sigma_floor_rel=0.5, alpha=0.03,
    signal_floor_abs={'pkt_size_mean':120.0,'port_div':2.0},
    signal_floor_rel={'pkt_size_mean':0.0,'port_div':0.0})))
perturb('alpha=0.10', lambda: DetectorConfig(baseline=BaselineConfig(
    min_samples=3, sigma_floor_rel=0.5, alpha=0.10,
    signal_floor_abs={'pkt_size_mean':120.0,'port_div':2.0},
    signal_floor_rel={'pkt_size_mean':0.0,'port_div':0.0})))
perturb('sigma_floor_rel=0.4', lambda: DetectorConfig(baseline=BaselineConfig(
    min_samples=3, sigma_floor_rel=0.4, alpha=0.05,
    signal_floor_abs={'pkt_size_mean':120.0,'port_div':2.0},
    signal_floor_rel={'pkt_size_mean':0.0,'port_div':0.0})))
perturb('sigma_floor_rel=0.6', lambda: DetectorConfig(baseline=BaselineConfig(
    min_samples=3, sigma_floor_rel=0.6, alpha=0.05,
    signal_floor_abs={'pkt_size_mean':120.0,'port_div':2.0},
    signal_floor_rel={'pkt_size_mean':0.0,'port_div':0.0})))
# fusion knobs
perturb('z_elevation=1.5', lambda: DetectorConfig(fusion=FusionConfig(z_elevation=1.5, z_saturate=6.0)))
perturb('z_elevation=2.5', lambda: DetectorConfig(fusion=FusionConfig(z_elevation=2.5, z_saturate=6.0)))
perturb('min_signals=3', lambda: DetectorConfig(fusion=FusionConfig(z_elevation=2.0, z_saturate=6.0, min_signals=3)))
# ladder knobs
perturb('score_high=0.25', lambda: DetectorConfig(ladder=LadderConfig(score_high=0.25, score_low=0.12, sustain_z=1.5, sustain_steps=12)))
perturb('score_high=0.40', lambda: DetectorConfig(ladder=LadderConfig(score_high=0.40, score_low=0.20, sustain_z=1.5, sustain_steps=12)))
perturb('sustain_steps=8', lambda: DetectorConfig(ladder=LadderConfig(score_high=0.30, score_low=0.15, sustain_z=1.5, sustain_steps=8)))
perturb('sustain_steps=20', lambda: DetectorConfig(ladder=LadderConfig(score_high=0.30, score_low=0.15, sustain_z=1.5, sustain_steps=20)))
# broad gate knobs
perturb('bw_frac_threshold=0.6', lambda: DetectorConfig(bw_frac_threshold=0.6))
perturb('bw_frac_threshold=0.4', lambda: DetectorConfig(bw_frac_threshold=0.4))

def inject(s):
    c = DetectorConfig(); c.broad_bw_score = s; return c
perturb('broad_bw_score=0.30', lambda: inject(0.30))
perturb('broad_bw_score=0.40', lambda: inject(0.40))
perturb('broad_bw_steps=2', lambda: DetectorConfig(broad_bw_steps=2))
perturb('broad_bw_steps=5', lambda: DetectorConfig(broad_bw_steps=5))