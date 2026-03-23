"""
models.py — Battery equivalent circuit models and parameter fitting.

Equivalent circuit: Rs + R1||C1 + R2||C2 (Extended Randles)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────
# Data transfer object
# ──────────────────────────────────────────────

@dataclass
class FitResult:
    Rs: float
    R1: float
    C1: float
    R2: float
    C2: float
    sigma_R1: float
    sigma_C1: float
    sigma_R2: float
    sigma_C2: float
    r2: float
    rmse_mv: float          # RMSE in millivolts
    tau1: float = field(init=False)
    tau2: float = field(init=False)
    f1: float = field(init=False)
    f2: float = field(init=False)

    def __post_init__(self):
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2
        self.f1 = 1.0 / (2.0 * math.pi * self.tau1) if self.tau1 > 0 else float("nan")
        self.f2 = 1.0 / (2.0 * math.pi * self.tau2) if self.tau2 > 0 else float("nan")


# ──────────────────────────────────────────────
# Time-domain voltage response models
# ──────────────────────────────────────────────

def voltage_response_2rc(
    t: np.ndarray,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    Vp2: float,
    Rs: float,
    I: float,
) -> np.ndarray:
    """Extended Randles time-domain response.

    V(t) = Vp2 + Rs*I + R1*I*(1 - exp(-t/τ1)) + R2*I*(1 - exp(-t/τ2))

    Rs, Vp2, I are fixed constants (already known before fitting).
    R1, C1, R2, C2 are the free parameters fitted by curve_fit.
    """
    tau1 = R1 * C1
    tau2 = R2 * C2
    return (
        Vp2
        + Rs * I
        + R1 * I * (1.0 - np.exp(-t / tau1))
        + R2 * I * (1.0 - np.exp(-t / tau2))
    )


def voltage_response_1rc(
    t: np.ndarray,
    R1: float,
    C1: float,
    Vp2: float,
    Rs: float,
    I: float,
) -> np.ndarray:
    """Simple Randles time-domain response (Rs + R1||C1 only)."""
    tau1 = R1 * C1
    return Vp2 + Rs * I + R1 * I * (1.0 - np.exp(-t / tau1))


# ──────────────────────────────────────────────
# Frequency-domain impedance model
# ──────────────────────────────────────────────

def impedance_2rc(
    f: np.ndarray,
    Rs: float,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
) -> np.ndarray:
    """Complex impedance Z(f) for Extended Randles circuit.

    Z(f) = Rs + R1/(1 + j·ω·R1·C1) + R2/(1 + j·ω·R2·C2)
    """
    omega = 2.0 * math.pi * np.asarray(f, dtype=float)
    Z1 = R1 / (1.0 + 1j * omega * R1 * C1)
    Z2 = R2 / (1.0 + 1j * omega * R2 * C2)
    return Rs + Z1 + Z2


# ──────────────────────────────────────────────
# Parameter fitting
# ──────────────────────────────────────────────

# Default initial values for 21700 5Ah cell
_P0_2RC = [0.005, 2.0, 0.010, 150.0]          # R1, C1, R2, C2
_LB_2RC = [1e-6,  1e-3, 1e-6,  1e-3]
_UB_2RC = [1.0,   500.0, 1.0,  2000.0]        # C2 upper = 2000 (not 100!)

_P0_1RC = [0.005, 2.0]
_LB_1RC = [1e-6,  1e-3]
_UB_1RC = [1.0,   500.0]


def fit_parameters(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    model: str = "extended",
    use_lmfit: bool = False,
) -> FitResult:
    """Fit equivalent circuit parameters to measured voltage transient.

    Parameters
    ----------
    t_fit   : time array offset so that t[0] == 0 (at p2)
    V_fit   : measured voltage array (same length as t_fit)
    Rs      : pre-calculated ohmic resistance (fixed, not fitted)
    I       : applied current in Amperes (positive)
    Vp2     : voltage at p2 (initial condition, fixed)
    model   : 'extended' (Rs+R1C1+R2C2) or 'simple' (Rs+R1C1)
    use_lmfit: use lmfit instead of scipy (richer uncertainty output)

    Returns
    -------
    FitResult with fitted params, 1-sigma errors, R², RMSE(mV)
    """
    if model == "simple":
        return _fit_1rc(t_fit, V_fit, Rs, I, Vp2)

    return _fit_2rc(t_fit, V_fit, Rs, I, Vp2, use_lmfit)


def _fit_2rc(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    use_lmfit: bool,
) -> FitResult:
    def model_fixed(t, R1, C1, R2, C2):
        return voltage_response_2rc(t, R1, C1, R2, C2, Vp2, Rs, I)

    if use_lmfit:
        try:
            import lmfit
        except ImportError:
            use_lmfit = False

    if use_lmfit:
        import lmfit
        params = lmfit.Parameters()
        for name, p0, lb, ub in zip(
            ["R1", "C1", "R2", "C2"], _P0_2RC, _LB_2RC, _UB_2RC
        ):
            params.add(name, value=p0, min=lb, max=ub)

        def residual(p):
            return model_fixed(t_fit, p["R1"], p["C1"], p["R2"], p["C2"]) - V_fit

        lm_result = lmfit.minimize(residual, params, method="least_squares")
        p = lm_result.params
        R1, C1 = p["R1"].value, p["C1"].value
        R2, C2 = p["R2"].value, p["C2"].value

        def _s(par):
            return par.stderr if par.stderr is not None else float("nan")

        sigma = [_s(p["R1"]), _s(p["C1"]), _s(p["R2"]), _s(p["C2"])]
        V_pred = model_fixed(t_fit, R1, C1, R2, C2)
    else:
        popt, pcov = curve_fit(
            model_fixed,
            t_fit,
            V_fit,
            p0=_P0_2RC,
            bounds=(_LB_2RC, _UB_2RC),
            method="trf",
            maxfev=10000,
        )
        R1, C1, R2, C2 = popt
        sigma = np.sqrt(np.diag(pcov)).tolist()
        V_pred = model_fixed(t_fit, *popt)

    return FitResult(
        Rs=Rs, R1=R1, C1=C1, R2=R2, C2=C2,
        sigma_R1=sigma[0], sigma_C1=sigma[1],
        sigma_R2=sigma[2], sigma_C2=sigma[3],
        r2=_r2(V_fit, V_pred),
        rmse_mv=_rmse_mv(V_fit, V_pred),
    )


def _fit_1rc(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
) -> FitResult:
    def model_fixed(t, R1, C1):
        return voltage_response_1rc(t, R1, C1, Vp2, Rs, I)

    popt, pcov = curve_fit(
        model_fixed,
        t_fit,
        V_fit,
        p0=_P0_1RC,
        bounds=(_LB_1RC, _UB_1RC),
        method="trf",
        maxfev=10000,
    )
    R1, C1 = popt
    sigma = np.sqrt(np.diag(pcov))
    V_pred = model_fixed(t_fit, *popt)

    return FitResult(
        Rs=Rs, R1=R1, C1=C1, R2=0.0, C2=0.0,
        sigma_R1=sigma[0], sigma_C1=sigma[1],
        sigma_R2=0.0, sigma_C2=0.0,
        r2=_r2(V_fit, V_pred),
        rmse_mv=_rmse_mv(V_fit, V_pred),
    )


# ──────────────────────────────────────────────
# Nyquist curve generation
# ──────────────────────────────────────────────

def compute_nyquist(
    Rs: float,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    f_range: tuple[float, float] = (0.01, 1e5),
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nyquist plot arrays from fitted parameters.

    Returns
    -------
    re_z     : Re(Z) array [Ohm]
    neg_im_z : -Im(Z) array [Ohm]
    """
    f_array = np.logspace(
        math.log10(f_range[0]), math.log10(f_range[1]), n_points
    )
    Z = impedance_2rc(f_array, Rs, R1, C1, R2, C2)
    return np.real(Z), -np.imag(Z)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _r2(y_meas: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_meas - y_pred) ** 2)
    ss_tot = np.sum((y_meas - np.mean(y_meas)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _rmse_mv(y_meas: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_meas - y_pred) ** 2)) * 1000.0)
