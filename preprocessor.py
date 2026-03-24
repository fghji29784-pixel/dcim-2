"""
preprocessor.py — Signal pre-processing for DCIM analysis.

Key fixes over prior known-buggy implementations:
  Bug 2: t is offset to p2=0 (not stored as absolute time)
  Bug 3: 5-second window derived from auto-detected dt (not 1000 fixed samples)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Pulse detection
# ──────────────────────────────────────────────

def detect_pulse(df: pd.DataFrame, threshold_fraction: float = 0.05) -> int:
    """Find the index where the current pulse begins.

    Looks for the first sample where |current_A| exceeds
    ``threshold_fraction`` × max(|current_A|).

    Parameters
    ----------
    df                 : DataFrame with 'current_A' column
    threshold_fraction : fraction of max current to use as onset threshold

    Returns
    -------
    int index of first above-threshold sample
    """
    i_abs = df["current_A"].abs()
    threshold = threshold_fraction * i_abs.max()
    mask = i_abs > threshold
    if not mask.any():
        raise ValueError(
            "No current pulse detected. Check that the file contains charge data "
            "and that the correct current unit (A/mA) is selected."
        )
    return int(mask.idxmax())


# ──────────────────────────────────────────────
# Critical point detection: p0, p1, p2
# ──────────────────────────────────────────────

def find_p0_p1_p2(
    df: pd.DataFrame,
    I_set: float,
) -> tuple[int, int, int]:
    """Locate the three critical points in the charge pulse.

    Definitions
    -----------
    p0 : last sample **before** the current pulse (rest, I ≈ 0)
    p1 : first sample (after pulse onset) where |I_set - I| / I_set < 50%
    p2 : first sample where |I_set - I| / I_set < 1%
         (relaxed to 5% if strict threshold is never met)

    Parameters
    ----------
    df    : DataFrame with 'current_A' column (standardised by loader.py)
    I_set : target / set-point current in Amperes (positive value)

    Returns
    -------
    (idx_p0, idx_p1, idx_p2) — integer row-label indices into df
    """
    pulse_start = detect_pulse(df)

    # p0: one sample before the pulse starts (rest state)
    idx_p0 = max(0, pulse_start - 1)

    # Search only from pulse_start onward to avoid pre-pulse false-positives
    search = df.iloc[pulse_start:].copy()
    rel_err = (I_set - search["current_A"]).abs() / abs(I_set)

    # p1: first where within 50% of I_set
    mask_p1 = rel_err < 0.50
    if not mask_p1.any():
        raise ValueError(
            "Cannot find p1: current never reaches 50% of I_set. "
            "Check I_set or data integrity."
        )
    idx_p1 = int(mask_p1.idxmax())

    # p2: first where within 1% (strict), fall back to 5%
    mask_p2_strict = rel_err < 0.01
    if mask_p2_strict.any():
        idx_p2 = int(mask_p2_strict.idxmax())
    else:
        mask_p2_loose = rel_err < 0.05
        if mask_p2_loose.any():
            idx_p2 = int(mask_p2_loose.idxmax())
        else:
            raise ValueError(
                "Cannot find p2: current never settles within 5% of I_set. "
                "Check I_set value or try setting it manually."
            )

    return idx_p0, idx_p1, idx_p2


# ──────────────────────────────────────────────
# Rs calculation
# ──────────────────────────────────────────────

def calculate_Rs(df: pd.DataFrame, idx_p0: int, idx_p1: int) -> float:
    """Compute ohmic resistance from instantaneous voltage/current jump.

    Rs = ΔV / ΔI  measured between p0 (rest) and p1 (first 50% point)

    Parameters
    ----------
    df              : DataFrame with 'voltage_V' and 'current_A'
    idx_p0, idx_p1  : integer indices (row labels) returned by find_p0_p1_p2

    Returns
    -------
    Rs in Ohms (positive for a normal ohmic drop)
    """
    dV = float(df.loc[idx_p1, "voltage_V"] - df.loc[idx_p0, "voltage_V"])
    dI = float(df.loc[idx_p1, "current_A"] - df.loc[idx_p0, "current_A"])
    if abs(dI) < 1e-9:
        raise ValueError(
            "ΔI ≈ 0 between p0 and p1; cannot compute Rs. "
            "Check that current unit conversion is correct."
        )
    return dV / dI


# ──────────────────────────────────────────────
# Fit data preparation
# ──────────────────────────────────────────────

def prepare_fit_data(
    df: pd.DataFrame,
    idx_p2: int,
    dt: float | None = None,
    window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract and prepare the voltage transient window for curve fitting.

    Uses time-based window selection (not sample-count-based) so that
    mixed-sampling-rate files (DCIM burst at 192 µs + normal CC data at
    0.1–1 s) are handled correctly.  dt is estimated from the local
    neighbourhood of p2, not the whole-file median, for the same reason.

    Parameters
    ----------
    df       : DataFrame with 'time_s' and 'voltage_V'
    idx_p2   : row-label index of p2 point
    dt       : sampling interval in seconds; auto-detected if None
    window_s : length of fitting window in seconds (default 5.0 s)

    Returns
    -------
    t_fit   : time array [s], starts at 0 (p2 = t=0)
    V_fit   : voltage array [V], same length as t_fit
    Vp2     : voltage at p2 [V]
    dt      : local sampling interval around p2 [s]
    """
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values

    # ── Row-position of p2 ────────────────────────────────────────────────
    pos_p2 = df.index.get_loc(idx_p2)
    t_p2_abs = float(time_all[pos_p2])

    # ── Time-based window selection ───────────────────────────────────────
    # Works correctly regardless of sampling rate changes in the file.
    t_end = t_p2_abs + window_s
    pos_end = int(np.searchsorted(time_all, t_end, side="right"))
    pos_end = min(pos_end, len(df))

    if pos_end <= pos_p2:
        raise ValueError(
            "p2 is at or beyond the end of the data. "
            "Cannot extract a fitting window."
        )

    t_window = time_all[pos_p2:pos_end]
    V_window = volt_all[pos_p2:pos_end]

    # ── Local dt around p2 ────────────────────────────────────────────────
    # Using the local neighbourhood avoids the median being skewed by
    # the slow-sampled portions of a mixed-rate file.
    if dt is None:
        lo = max(0, pos_p2 - 5)
        hi = min(len(df), pos_p2 + 6)
        local_diffs = np.diff(time_all[lo:hi])
        positive_diffs = local_diffs[local_diffs > 0]
        if len(positive_diffs) > 0:
            dt = float(np.median(positive_diffs))
        else:
            all_diffs = np.diff(time_all)
            positive_all = all_diffs[all_diffs > 0]
            dt = float(np.median(positive_all)) if len(positive_all) > 0 else 1.0
        if dt <= 0:
            raise ValueError(
                f"Auto-detected dt = {dt:.6g} s is non-positive. "
                "Check that the time column is in seconds and monotonically increasing."
            )

    Vp2 = float(V_window[0])

    # ── t offset: 0-based from p2 ─────────────────────────────────────────
    t_fit = t_window - t_p2_abs   # t_fit[0] == 0.0

    return t_fit, V_window.copy(), Vp2, dt


# ──────────────────────────────────────────────
# Convenience: auto-detect I_set from data
# ──────────────────────────────────────────────

def detect_I_set(df: pd.DataFrame) -> float:
    """Estimate the target current as the 95th percentile of |current_A|.

    Using the 95th percentile (rather than the maximum) guards against
    transient spikes inflating the estimate.
    """
    return float(np.percentile(df["current_A"].abs(), 95))
