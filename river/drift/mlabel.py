from __future__ import annotations

from collections.abc import Iterable

from river.base import DriftDetector

from .adwin_c import AdaptiveWindowing


class MLABEL(DriftDetector):
    """Multi-label Drift Detector using ADWIN for each label independently.

    Parameters
    ----------
    delta
        Significance level for ADWIN.
    clock
        Frequency for ADWIN to check for changes.
    max_buckets
        Max number of buckets per ADWIN detector.
    min_window_length
        Minimum window size to be considered for drift detection.
    grace_period
        Number of initial samples to wait before checking for drift.
    """

    def __init__(self, delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10):
        super().__init__()
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self._helpers: dict[int, AdaptiveWindowing] = {}
        self._drift_detected = {}

    def _init_helper(self, label_index: int):
        self._helpers[label_index] = AdaptiveWindowing(
            delta=self.delta,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )
        self._drift_detected[label_index] = False

    def update(self, x: Iterable[int] | list[int]):
        """Update all label-wise detectors with new label values (0 or 1 per label).

        Parameters
        ----------
        x : list or iterable
            A list of binary values (0/1) for each label.
        """
        for i, xi in enumerate(x):
            if i not in self._helpers:
                self._init_helper(i)

            if self._drift_detected[i]:
                self._init_helper(i)

            self._drift_detected[i] = self._helpers[i].update(xi)

        return self

    @property
    def drift_detected(self) -> dict[int, bool]:
        return self._drift_detected
