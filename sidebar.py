"""
sidebar.py — Streamlit sidebar UI components for DCIM analyzer.

All functions are pure UI; they return values and do not mutate session state.
"""

from __future__ import annotations

import streamlit as st


def render_file_upload() -> tuple:
    """Render file upload widgets.

    Returns
    -------
    (charge_file, eis_file) — UploadedFile or None for each
    """
    st.subheader("Data Files")
    charge_file = st.file_uploader(
        "Charge data (BioLogic GCPL — .mpt / .csv / .txt)",
        type=["mpt", "csv", "txt"],
        key="charge_file",
    )
    eis_file = st.file_uploader(
        "EIS data (BioLogic GEIS — optional)",
        type=["mpt", "csv", "txt"],
        key="eis_file",
        help="Upload measured EIS data to overlay on the Nyquist plot.",
    )
    return charge_file, eis_file


def render_current_unit() -> str:
    """Render current-unit selector.

    Returns
    -------
    'mA' or 'A'
    """
    st.subheader("Data Settings")
    return st.selectbox(
        "Current unit in file",
        options=["mA", "A"],
        index=0,
        help="BioLogic GCPL files typically export current in mA. "
             "Select 'mA' to auto-convert to Amperes.",
    )


def render_model_selector() -> str:
    """Render equivalent circuit model selector.

    Returns
    -------
    'extended' or 'simple'
    """
    choice = st.selectbox(
        "Equivalent circuit model",
        options=[
            "Extended Randles  (Rs + R1C1 + R2C2)  ← recommended",
            "Simple Randles    (Rs + R1C1)",
        ],
        index=0,
    )
    return "simple" if "Simple" in choice else "extended"


def render_manual_range() -> tuple[int | None, float]:
    """Render optional manual override for p2 index and fit window.

    Returns
    -------
    (p2_override, window_s)
    p2_override : int if manually set, else None
    window_s    : fit window length in seconds (default 5.0)
    """
    st.subheader("Advanced Options")
    use_manual = st.checkbox(
        "Override p2 index manually",
        value=False,
        help="Use this if the automatic p2 detection fails.",
    )
    p2_override = None
    if use_manual:
        p2_override = st.number_input(
            "p2 index (row number in loaded data)",
            min_value=0,
            value=0,
            step=1,
        )
        p2_override = int(p2_override)

    window_s = st.number_input(
        "Fit window (seconds from p2)",
        min_value=0.5,
        max_value=60.0,
        value=5.0,
        step=0.5,
        help="Length of voltage transient used for fitting. "
             "Must cover at least τ2 to resolve C2.",
    )
    return p2_override, float(window_s)


def render_fit_engine() -> bool:
    """Render fitting engine selector.

    Returns
    -------
    use_lmfit : bool (True = lmfit, False = scipy)
    """
    engine = st.selectbox(
        "Fitting engine",
        options=[
            "scipy.optimize.curve_fit  (fast, standard)",
            "lmfit  (slower, better uncertainty estimates)",
        ],
        index=0,
    )
    return "lmfit" in engine
