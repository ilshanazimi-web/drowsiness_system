"""
smoothing.py - temporal smoothing to reduce flicker
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque

from utils.config import CONFIG


class TemporalSmoother:
    def __init__(self, config: type[CONFIG.smoothing] | None=None) -> None:
        self._config = config or CONFIG.smoothing
        self._states = CONFIG.states
        self._history: Deque[str] = deque(maxlen=self._config.window_size)
        self._current_stable_state: str = self._states.ALERT
        self._candidate_state: str = self._states.ALERT
        self._candidate_count: int = 0

    def update(self, raw_label: str) -> str:
        self._history.append(raw_label)
        majority_label = self._majority_vote()

        if majority_label == self._current_stable_state:
            self._candidate_state =majority_label
            self._candidate_count =0
            return self._current_stable_state

        if majority_label == self._candidate_state:
            self._candidate_count+= 1
        else:
            self._candidate_state = majority_label
            self._candidate_count = 1

        if self._candidate_count >= self._config.min_state_persistence:
            self._current_stable_state = self._candidate_state
            self._candidate_count = 0

        return self._current_stable_state

    def _majority_vote(self) ->str:
        if not self._history:
            return self._states.ALERT
        counts = Counter(self._history)
        return counts.most_common(1)[0][0]

    @property
    def current_state(self) -> str:
        return self._current_stable_state

    def reset(self) -> None:
        self._history.clear()
        self._current_stable_state = self._states.ALERT
        self._candidate_state = self._states.ALERT
        self._candidate_count= 0
