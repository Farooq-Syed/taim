"""Unit tests for Phase 4: graduated response ladder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ladder import ACTIONS, STAGE_NAMES, LadderConfig, ResponseLadder


def test_fresh_device_is_normal():
    ladder = ResponseLadder()
    assert ladder.stage_of(7) == 0


def test_full_escalation_to_drop():
    ladder = ResponseLadder(LadderConfig(escalate_steps=2, deescalate_steps=4))
    stages = []
    for _ in range(9):  # 9 high steps -> stage 0..4 then hold at 4
        stage, action = ladder.step(0, 0.9, fired=True)
        stages.append(stage)
    assert stages == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert action == "drop_deauth"


def test_sporadic_high_scores_do_not_escalate():
    ladder = ResponseLadder(LadderConfig(escalate_steps=3))
    seq = [0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9]  # no run of 3 consecutive highs
    stage = 0
    for s in seq:
        stage, _ = ladder.step(0, s, fired=True)
    assert stage == 0


def test_deescalation_after_recovery():
    ladder = ResponseLadder(LadderConfig(escalate_steps=1, deescalate_steps=3))
    for _ in range(5):
        ladder.step(0, 0.9, fired=True)
    assert ladder.stage_of(0) == 4
    for _ in range(3):
        stage, _ = ladder.step(0, 0.05, fired=False)
    assert stage == 3
    for _ in range(3):
        stage, _ = ladder.step(0, 0.05, fired=False)
    assert stage == 2


def test_hysteresis_band_holds_state():
    ladder = ResponseLadder(LadderConfig(escalate_steps=2, deescalate_steps=2))
    for _ in range(4):
        ladder.step(0, 0.9, fired=True)  # up to stage 2
    # mid-band scores (between low and high) must not move the stage
    for _ in range(10):
        stage, _ = ladder.step(0, 0.35, fired=True)
        assert stage == 2


def test_zero_score_deescalates():
    ladder = ResponseLadder(LadderConfig(escalate_steps=1, deescalate_steps=2))
    for _ in range(3):
        ladder.step(0, 0.9, fired=True)
    assert ladder.stage_of(0) == 3
    for _ in range(2):
        stage, _ = ladder.step(0, 0.0, fired=False)
    assert stage == 2


def test_devices_are_independent():
    ladder = ResponseLadder(LadderConfig(escalate_steps=2))
    for _ in range(8):
        ladder.step(0, 0.9, fired=True)   # device 0 escalates to stage 4
        ladder.step(1, 0.0, fired=False)  # device 1 stays normal
    assert ladder.stage_of(0) == 4
    assert ladder.stage_of(1) == 0


def test_reset():
    ladder = ResponseLadder()
    for _ in range(4):
        ladder.step(0, 0.9, fired=True)
    assert ladder.stage_of(0) > 0
    ladder.reset(0)
    assert ladder.stage_of(0) == 0


def test_stage_names_and_actions_consistent():
    assert len(STAGE_NAMES) == len(ACTIONS) == 5
    assert ACTIONS[0] == "allow"
    assert ACTIONS[4] == "drop_deauth"


def test_sustain_path_escalates_on_same_signal():
    """A single persistently-anomalous metric (low-and-slow) must escalate
    even when the fused score stays suppressed."""
    ladder = ResponseLadder(LadderConfig(sustain_z=2.0, sustain_steps=6))
    for _ in range(6):
        stage, _ = ladder.step(0, score=0.0, fired=False, max_z=3.0, max_z_signal="bw")
    assert stage == 1


def test_sustain_path_requires_same_signal():
    """Flickering across different signals must NOT accumulate the streak."""
    ladder = ResponseLadder(LadderConfig(sustain_z=2.0, sustain_steps=6))
    signals = ["bw", "conn", "req", "port"]
    for i in range(12):
        stage, _ = ladder.step(0, score=0.0, fired=False, max_z=3.0,
                               max_z_signal=signals[i % len(signals)])
        assert stage == 0  # signal keeps changing -> streak never builds


def test_sustain_path_requires_consecutive():
    ladder = ResponseLadder(LadderConfig(sustain_z=2.0, sustain_steps=6))
    for i in range(10):  # 5 bw steps, interruption, then 4 bw steps
        if i == 5:
            ladder.step(0, score=0.0, fired=False, max_z=0.5, max_z_signal=None)
        stage, _ = ladder.step(0, score=0.0, fired=False, max_z=3.0, max_z_signal="bw")
    assert stage == 0  # never reached 6 consecutive after the interruption


def test_sustain_and_fused_paths_combine():
    ladder = ResponseLadder(LadderConfig(sustain_z=2.0, sustain_steps=4))
    # both paths fire -> escalate faster than either alone
    for _ in range(3):
        stage, _ = ladder.step(0, score=0.9, fired=True, max_z=5.0, max_z_signal="bw")
    assert stage >= 1