"""Phase 4: Graduated response ladder.

Per-device state machine that turns the continuous suspicion score into a
discrete mitigation action. Escalation is progressive (watch -> soft cap ->
hard cap -> drop) and de-escalation is graceful once the device recovers.

The two thresholds (score_high / score_low) form a hysteresis band so a score
hovering around one threshold doesn't make the device oscillate between stages.
"""
from __future__ import annotations

from dataclasses import dataclass

STAGE_NAMES = ["normal", "watch", "soft_cap", "hard_cap", "drop"]
ACTIONS = {
    0: "allow",
    1: "watch",
    2: "soft_cap_70pct",
    3: "hard_cap_30pct",
    4: "drop_deauth",
}


@dataclass
class LadderConfig:
    score_high: float = 0.5         # above this -> counts toward escalation
    score_low: float = 0.2          # below this -> counts toward de-escalation
    escalate_steps: int = 2         # consecutive high steps to advance a stage
    deescalate_steps: int = 4       # consecutive low steps to retreat a stage
    max_stage: int = 4
    # sustained single-signal path: if the same metric stays above this |z|
    # for this many consecutive steps, escalate. Catches low-and-slow
    # distributed attacks that never light up two signals at once.
    sustain_z: float = 2.0
    sustain_steps: int = 15
    # if True, the streak only accumulates while the SAME signal is the max;
    # if False, only the z level matters (more robust when two signals are
    # both persistently elevated, e.g. an aggregate under a low-and-slow).
    sustain_same_signal: bool = True


@dataclass
class DeviceState:
    stage: int = 0
    high_streak: int = 0
    low_streak: int = 0
    sustain_streak: int = 0
    sustain_signal: str | None = None


class ResponseLadder:
    def __init__(self, config: LadderConfig | None = None) -> None:
        self.config = config or LadderConfig()
        self.states: dict[int, DeviceState] = {}

    def _state(self, device_id: int) -> DeviceState:
        return self.states.setdefault(device_id, DeviceState())

    def stage_of(self, device_id: int) -> int:
        return self._state(device_id).stage

    def step(
        self,
        device_id: int,
        score: float,
        fired: bool,
        max_z: float = 0.0,
        max_z_signal: str | None = None,
    ) -> tuple[int, str]:
        """Advance the device's state machine. Returns (stage, action).

        Two independent escalation paths:
          * fused  - score > score_high for escalate_steps (fast, multi-signal)
          * sustain- the SAME single metric above sustain_z for sustain_steps
                     (slow, catches low-and-slow)
        """
        cfg = self.config
        st = self._state(device_id)

        # ---- fused path ----
        if score > cfg.score_high:
            st.high_streak += 1
            st.low_streak = 0
            if st.high_streak >= cfg.escalate_steps and st.stage < cfg.max_stage:
                st.stage += 1
                st.high_streak = 0
        elif score < cfg.score_low:
            st.low_streak += 1
            st.high_streak = 0
            if st.low_streak >= cfg.deescalate_steps and st.stage > 0:
                st.stage -= 1
                st.low_streak = 0
        else:
            st.high_streak = 0
            st.low_streak = 0

        # ---- sustained single-signal path ----
        if max_z >= cfg.sustain_z:
            if cfg.sustain_same_signal:
                if max_z_signal == st.sustain_signal:
                    st.sustain_streak += 1
                else:
                    st.sustain_signal = max_z_signal
                    st.sustain_streak = 1
            else:
                st.sustain_streak += 1
            if st.sustain_streak >= cfg.sustain_steps and st.stage < cfg.max_stage:
                st.stage += 1
                st.sustain_streak = 0
        else:
            st.sustain_streak = 0
            st.sustain_signal = None

        return st.stage, ACTIONS[st.stage]

    def reset(self, device_id: int | None = None) -> None:
        if device_id is None:
            self.states.clear()
        else:
            self.states.pop(device_id, None)

    def stage_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {i: 0 for i in range(5)}
        for st in self.states.values():
            dist[st.stage] += 1
        return dist
