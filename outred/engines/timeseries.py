# outred/engines/timeseries.py
# Time-series anomaly detection engine (V2 — planned).
#
# STATUS: Not yet implemented. When a CSV with datetime columns is uploaded,
# the dispatcher passes data through unchanged with zero anomaly scores.
# This module exists as a placeholder for the V2 implementation.
#
# Planned approach:
#   - Seasonal decomposition (STL) for trend/residual separation
#   - Residual-based scoring using IQR or Z-score
#   - Change-point detection for structural breaks
#
# Until implemented, the dispatcher logs a clear warning when datetime
# columns are detected so users know why no outliers were found.

import logging

logger = logging.getLogger(__name__)


def detect_timeseries_outliers(*args, **kwargs):
    """
    Placeholder for V2 time-series anomaly detection.

    Raises NotImplementedError with a clear message so callers know
    this feature is not yet available.
    """
    raise NotImplementedError(
        "Time-series anomaly detection is planned for V2. "
        "Currently, datasets with datetime columns are passed through "
        "with zero anomaly scores. Only numeric and categorical "
        "outlier detection is active."
    )
