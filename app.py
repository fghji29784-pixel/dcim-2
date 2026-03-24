"""
app.py — DCIM Battery Analyzer — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import traceback

import numpy as np
import streamlit as st

from loader import load_charge_data, load_eis_data
from preprocessor import find_p0_p1_p2, calculate_Rs, prepare_fit_data, detect_I_set
from models import fit_parameters, compute_nyquist, voltage_response_2rc, voltage_response_1rc
from plotter import plot_raw_data, plot_fit_result, plot_nyquist
from exporter import export_results_excel, export_report_text
from sidebar import (
    render_file_upload,
    render_current_unit,
    render_model_selector,
    render_manual_range,
    render_fit_engine,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="DCIM Battery Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────

_STATE_KEYS = [
    "df_charge", "df_eis",
    "idx_p0", "idx_p1", "idx_p2",
    "Rs", "I_set",
    "t_fit", "V_fit", "V_pred", "Vp2", "dt",
    "fit_result",
    "re_z", "neg_im_z",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("🔋 DCIM Analyzer")
    st.markdown("---")

    charge_file, eis_file = render_file_upload()
    current_unit = render_current_unit()
    model_choice = render_model_selector()
    p2_override, window_s = render_manual_range()
    use_lmfit = render_fit_engine()

    st.markdown("---")
    run_button = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────

st.title("DCIM Battery Impedance Analyzer")
st.caption(
    "DC Impedance Measurement — extracts Rs, R1, C1, R2, C2 from charge transients "
    "and reconstructs the Nyquist plot without EIS hardware."
)

tab_raw, tab_fit, tab_nyquist, tab_export = st.tabs([
    "📈 Raw Data", "🔧 Fit Result", "🔵 Nyquist Plot", "📥 Export"
])

# ──────────────────────────────────────────────
# Analysis pipeline (runs when button is pressed)
# ──────────────────────────────────────────────

if run_button:
    if charge_file is None:
        st.error("Please upload a charge data file first.")
        st.stop()

    # Reset state
    for k in _STATE_KEYS:
        st.session_state[k] = None

    try:
        # ── Step 1: Load charge data ──────────────────────────────────────
        with st.spinner("Loading charge data…"):
            charge_file.seek(0)
            df = load_charge_data(charge_file, current_unit=current_unit)
            st.session_state.df_charge = df

        # ── Step 2: Load EIS data (optional) ─────────────────────────────
        if eis_file is not None:
            with st.spinner("Loading EIS data…"):
                eis_file.seek(0)
                df_eis = load_eis_data(eis_file)
                st.session_state.df_eis = df_eis

        # ── Step 3: Detect I_set and critical points ──────────────────────
        with st.spinner("Detecting p0, p1, p2…"):
            I_set = detect_I_set(df)
            idx_p0, idx_p1, idx_p2 = find_p0_p1_p2(df, I_set)
            if p2_override is not None:
                # Map positional override to df index label
                idx_p2 = df.index[p2_override]

            st.session_state.I_set = I_set
            st.session_state.idx_p0 = idx_p0
            st.session_state.idx_p1 = idx_p1
            st.session_state.idx_p2 = idx_p2

        # ── Step 4: Calculate Rs ──────────────────────────────────────────
        with st.spinner("Calculating Rs…"):
            Rs = calculate_Rs(df, idx_p0, idx_p1)
            st.session_state.Rs = Rs

        # ── Step 5: Prepare fitting data ──────────────────────────────────
        with st.spinner("Preparing fit data…"):
            t_fit, V_fit, Vp2, dt = prepare_fit_data(
                df, idx_p2, window_s=window_s
            )
            st.session_state.t_fit = t_fit
            st.session_state.V_fit = V_fit
            st.session_state.Vp2 = Vp2
            st.session_state.dt = dt

        # ── Step 6: Fit equivalent circuit model ─────────────────────────
        with st.spinner("Fitting model parameters…"):
            result = fit_parameters(
                t_fit, V_fit,
                Rs=Rs,
                I=I_set,
                Vp2=Vp2,
                model=model_choice,
                use_lmfit=use_lmfit,
            )
            st.session_state.fit_result = result

        # ── Step 7: Compute predicted voltage + Nyquist curve ─────────────
        with st.spinner("Computing Nyquist curve…"):
            if model_choice == "simple":
                V_pred = voltage_response_1rc(t_fit, result.R1, result.C1, Vp2, Rs, I_set)
            else:
                V_pred = voltage_response_2rc(
                    t_fit, result.R1, result.C1, result.R2, result.C2, Vp2, Rs, I_set
                )
            st.session_state.V_pred = V_pred

            re_z, neg_im_z = compute_nyquist(
                result.Rs, result.R1, result.C1, result.R2, result.C2
            )
            st.session_state.re_z = re_z
            st.session_state.neg_im_z = neg_im_z

        st.success("Analysis complete!")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# ──────────────────────────────────────────────
# Tab: Raw Data
# ──────────────────────────────────────────────

with tab_raw:
    if st.session_state.df_charge is not None and st.session_state.idx_p2 is not None:
        df = st.session_state.df_charge
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("p0 index", st.session_state.idx_p0)
        col2.metric("p1 index", st.session_state.idx_p1)
        col3.metric("p2 index", st.session_state.idx_p2)
        col4.metric("Rs", f"{st.session_state.Rs * 1000:.3f} mΩ")

        col_a, col_b = st.columns(2)
        col_a.metric("I_set", f"{st.session_state.I_set * 1000:.1f} mA")
        col_b.metric("dt", f"{st.session_state.dt * 1000:.3f} ms")

        fig = plot_raw_data(
            df,
            st.session_state.idx_p0,
            st.session_state.idx_p1,
            st.session_state.idx_p2,
        )
        st.pyplot(fig)

        with st.expander("View raw data table (first 200 rows)"):
            st.dataframe(df.head(200))
    else:
        st.info("Upload a charge data file and click **Run Analysis** to begin.")

# ──────────────────────────────────────────────
# Tab: Fit Result
# ──────────────────────────────────────────────

with tab_fit:
    result = st.session_state.fit_result
    if result is not None:
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rs",   f"{result.Rs * 1000:.3f} mΩ")
        col2.metric("R1",   f"{result.R1 * 1000:.3f} mΩ")
        col3.metric("R2",   f"{result.R2 * 1000:.3f} mΩ")
        col4.metric("R²",   f"{result.r2:.5f}")
        col5.metric("RMSE", f"{result.rmse_mv:.3f} mV")

        fig = plot_fit_result(
            st.session_state.t_fit,
            st.session_state.V_fit,
            st.session_state.V_pred,
            result,
        )
        st.pyplot(fig)

        # Full parameter table
        import pandas as pd
        param_df = pd.DataFrame({
            "Parameter": ["Rs", "R1", "C1", "R2", "C2", "τ1", "τ2", "f1", "f2", "R²", "RMSE"],
            "Value": [
                f"{result.Rs * 1000:.4f} mΩ",
                f"{result.R1 * 1000:.4f} mΩ",
                f"{result.C1:.4f} F",
                f"{result.R2 * 1000:.4f} mΩ",
                f"{result.C2:.4f} F",
                f"{result.tau1 * 1000:.3f} ms",
                f"{result.tau2:.4f} s",
                f"{result.f1:.3f} Hz",
                f"{result.f2:.5f} Hz",
                f"{result.r2:.6f}",
                f"{result.rmse_mv:.4f} mV",
            ],
            "±1σ": [
                "—",
                f"{result.sigma_R1 * 1000:.4f} mΩ",
                f"{result.sigma_C1:.4f} F",
                f"{result.sigma_R2 * 1000:.4f} mΩ",
                f"{result.sigma_C2:.4f} F",
                "—", "—", "—", "—", "—", "—",
            ],
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the analysis to see fitting results.")

# ──────────────────────────────────────────────
# Tab: Nyquist Plot
# ──────────────────────────────────────────────

with tab_nyquist:
    if st.session_state.re_z is not None:
        fig = plot_nyquist(
            st.session_state.re_z,
            st.session_state.neg_im_z,
            eis_df=st.session_state.df_eis,
            result=st.session_state.fit_result,
        )
        st.pyplot(fig)

        # ── EIS vs DCIM comparison ────────────────────────────────────────
        if st.session_state.df_eis is not None:
            import pandas as pd
            df_eis = st.session_state.df_eis
            result = st.session_state.fit_result

            rs_eis = float(df_eis["re_z"].min()) * 1000       # high-freq intercept [mΩ]
            rs_dcim = result.Rs * 1000

            # EIS arc peak: frequency where -Im(Z) is maximum → R_arc ≈ Re(Z_peak) - Rs
            idx_peak = df_eis["neg_im_z"].idxmax()
            re_at_peak_eis = float(df_eis.loc[idx_peak, "re_z"]) * 1000
            r_arc_eis = re_at_peak_eis - rs_eis               # R1+R2 from EIS arc width
            r_arc_dcim = (result.R1 + result.R2) * 1000

            st.subheader("EIS vs DCIM Comparison")
            cmp_df = pd.DataFrame({
                "Parameter": ["Rs (Ω resistance)", "R1+R2 (arc width, approx)"],
                "EIS [mΩ]":  [f"{rs_eis:.3f}", f"{r_arc_eis:.3f}"],
                "DCIM [mΩ]": [f"{rs_dcim:.3f}", f"{r_arc_dcim:.3f}"],
                "Diff [mΩ]": [
                    f"{rs_dcim - rs_eis:+.3f}",
                    f"{r_arc_dcim - r_arc_eis:+.3f}",
                ],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            st.caption(
                "EIS Rs = high-frequency Re(Z) intercept (minimum Re(Z)). "
                "EIS arc width = Re(Z) at -Im(Z) peak − Rs. "
                "These are geometric estimates; for precise EIS fitting, "
                "use dedicated EIS analysis software."
            )

        with st.expander("View Nyquist data table"):
            import pandas as pd
            nyq_df = pd.DataFrame({
                "Re(Z) [mΩ]":   st.session_state.re_z * 1000,
                "-Im(Z) [mΩ]":  st.session_state.neg_im_z * 1000,
            })
            st.dataframe(nyq_df, use_container_width=True)
    else:
        st.info("Run the analysis to see the Nyquist plot.")

# ──────────────────────────────────────────────
# Tab: Export
# ──────────────────────────────────────────────

with tab_export:
    result = st.session_state.fit_result
    if result is not None:
        st.subheader("Download Results")

        # Excel
        excel_bytes = export_results_excel(
            result,
            st.session_state.t_fit,
            st.session_state.V_fit,
            st.session_state.V_pred,
            (st.session_state.re_z, st.session_state.neg_im_z),
        )
        st.download_button(
            label="📊 Download Excel Report (.xlsx)",
            data=excel_bytes,
            file_name="dcim_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Text report
        report_str = export_report_text(result)
        st.download_button(
            label="📄 Download Text Report (.txt)",
            data=report_str,
            file_name="dcim_report.txt",
            mime="text/plain",
        )

        st.subheader("Report Preview")
        st.code(report_str, language=None)
    else:
        st.info("Run the analysis to enable exports.")
