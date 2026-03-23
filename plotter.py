"""
plotter.py — Matplotlib figure generators for DCIM analysis.

All functions return matplotlib.figure.Figure objects for use with
st.pyplot() in the Streamlit app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from models import FitResult

matplotlib.use("Agg")  # non-interactive backend for Streamlit


# ──────────────────────────────────────────────
# Graph 1 — Raw data with p0/p1/p2 markers
# ──────────────────────────────────────────────

def plot_raw_data(
    df: pd.DataFrame,
    idx_p0: int,
    idx_p1: int,
    idx_p2: int,
) -> Figure:
    """Two-panel figure: voltage (top) and current (bottom) vs time.

    Vertical markers at p0 (gray dashed), p1 (orange dotted), p2 (red solid).
    """
    t = df["time_s"].values
    V = df["voltage_V"].values
    I = df["current_A"].values * 1000.0  # display in mA

    t_p0 = df.loc[idx_p0, "time_s"]
    t_p1 = df.loc[idx_p1, "time_s"]
    t_p2 = df.loc[idx_p2, "time_s"]

    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Voltage panel
    ax_v.plot(t, V, color="#2196F3", linewidth=0.8, label="Voltage")
    _add_pmarkers(ax_v, t_p0, t_p1, t_p2)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.legend(loc="upper left", fontsize=8)
    ax_v.grid(True, alpha=0.3)

    # Current panel
    ax_i.plot(t, I, color="#FF5722", linewidth=0.8, label="Current")
    _add_pmarkers(ax_i, t_p0, t_p1, t_p2)
    ax_i.set_ylabel("Current (mA)")
    ax_i.set_xlabel("Time (s)")
    ax_i.legend(loc="upper left", fontsize=8)
    ax_i.grid(True, alpha=0.3)

    # Shared legend for markers (add to bottom panel)
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], color="gray",   linestyle="--", label=f"p0 (rest)      t={t_p0:.4f} s"),
        Line2D([0], [0], color="orange", linestyle=":",  label=f"p1 (50% I_set) t={t_p1:.4f} s"),
        Line2D([0], [0], color="red",    linestyle="-",  label=f"p2 (settled)   t={t_p2:.4f} s"),
    ]
    ax_i.legend(handles=marker_handles, loc="lower right", fontsize=7)

    fig.suptitle("Raw Charge Data — Pulse Detection", fontsize=11, fontweight="bold")
    return fig


def _add_pmarkers(ax, t_p0: float, t_p1: float, t_p2: float):
    ax.axvline(t_p0, color="gray",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(t_p1, color="orange", linestyle=":",  linewidth=1.2, alpha=0.9)
    ax.axvline(t_p2, color="red",    linestyle="-",  linewidth=1.2, alpha=0.9)


# ──────────────────────────────────────────────
# Graph 2 — Fitting result with residuals
# ──────────────────────────────────────────────

def plot_fit_result(
    t_fit: np.ndarray,
    V_meas: np.ndarray,
    V_pred: np.ndarray,
    result: FitResult,
) -> Figure:
    """Two-panel figure: measured vs fitted voltage (top), residuals in mV (bottom).

    A parameter table is annotated as text on the right side of the top panel.
    """
    t_ms = t_fit * 1000.0       # convert to milliseconds for display
    residuals_mv = (V_meas - V_pred) * 1000.0

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3, 1],
                           height_ratios=[3, 1],
                           hspace=0.08, wspace=0.05)

    ax_fit = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[:, 1])

    # ── Fit overlay ──────────────────────────────────────────────────────
    ax_fit.plot(t_ms, V_meas * 1000, color="#2196F3", linewidth=1.0, label="Measured")
    ax_fit.plot(t_ms, V_pred * 1000, color="#F44336", linewidth=1.5,
                linestyle="--", label="Fitted")
    ax_fit.set_ylabel("Voltage (mV)")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.tick_params(labelbottom=False)
    ax_fit.set_title(
        f"Voltage Transient Fit  (R² = {result.r2:.5f},  RMSE = {result.rmse_mv:.3f} mV)",
        fontsize=10,
    )

    # ── Residuals ─────────────────────────────────────────────────────────
    ax_res.plot(t_ms, residuals_mv, color="#9C27B0", linewidth=0.8)
    ax_res.axhline(0, color="black", linewidth=0.6)
    ax_res.set_ylabel("Residual\n(mV)", fontsize=8)
    ax_res.set_xlabel("Time from p2 (ms)")
    ax_res.grid(True, alpha=0.3)

    # ── Parameter table ───────────────────────────────────────────────────
    ax_tbl.axis("off")
    table_data = [
        ["Parameter", "Value", "±1σ"],
        ["Rs",        f"{result.Rs * 1000:.3f} mΩ",  "—"],
        ["R1",        f"{result.R1 * 1000:.3f} mΩ",  f"{result.sigma_R1 * 1000:.3f} mΩ"],
        ["C1",        f"{result.C1:.3f} F",           f"{result.sigma_C1:.3f} F"],
        ["R2",        f"{result.R2 * 1000:.3f} mΩ",  f"{result.sigma_R2 * 1000:.3f} mΩ"],
        ["C2",        f"{result.C2:.2f} F",           f"{result.sigma_C2:.2f} F"],
        ["τ1",        f"{result.tau1 * 1000:.2f} ms", "—"],
        ["τ2",        f"{result.tau2:.3f} s",         "—"],
        ["f1",        f"{result.f1:.2f} Hz",          "—"],
        ["f2",        f"{result.f2:.4f} Hz",          "—"],
        ["R²",        f"{result.r2:.5f}",             "—"],
        ["RMSE",      f"{result.rmse_mv:.3f} mV",     "—"],
    ]
    tbl = ax_tbl.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    # Style header row
    for j in range(3):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("DCIM Parameter Fitting Result", fontsize=11, fontweight="bold")
    return fig


# ──────────────────────────────────────────────
# Graph 3 — Nyquist plot
# ──────────────────────────────────────────────

def plot_nyquist(
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
    eis_df: pd.DataFrame | None = None,
    result: FitResult | None = None,
) -> Figure:
    """Single-panel Nyquist plot.

    Parameters
    ----------
    re_z, neg_im_z : DCIM model curve arrays [Ohm]
    eis_df         : optional DataFrame with 're_z' and 'neg_im_z' columns
                     (measured EIS overlay)
    result         : optional FitResult for annotating Rs, peak frequencies
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # DCIM model curve
    ax.plot(re_z * 1000, neg_im_z * 1000,
            color="#2196F3", linewidth=1.8, label="DCIM model", zorder=3)

    # EIS measured overlay (optional)
    if eis_df is not None and len(eis_df) > 0:
        ax.scatter(
            eis_df["re_z"] * 1000,
            eis_df["neg_im_z"] * 1000,
            color="#F44336", s=18, zorder=4,
            label="EIS measured", marker="o",
        )

    # Annotate Rs on x-axis
    if result is not None:
        rs_mohm = result.Rs * 1000
        ax.axvline(rs_mohm, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"Rs={rs_mohm:.2f} mΩ",
            xy=(rs_mohm, 0),
            xytext=(rs_mohm + 0.3, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] != 0 else 0.1),
            fontsize=7,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
        )

    ax.set_xlabel("Re(Z) (mΩ)", fontsize=10)
    ax.set_ylabel("-Im(Z) (mΩ)", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Nyquist Plot — DCIM Reconstruction", fontsize=11, fontweight="bold")

    # Ensure y-axis starts at or below 0 (capacitive arcs are in positive quadrant)
    ymin, ymax = ax.get_ylim()
    if ymin > 0:
        ax.set_ylim(bottom=-0.02 * ymax)

    fig.tight_layout()
    return fig
