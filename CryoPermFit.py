import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# =========================
# Experimental data
# =========================
expdata = np.array([
    [267.99, 181.84],
    [267.15, 178.1667916],
    [266.15, 169.1173019],
    [265.15, 165.6389179],
    [264.15, 158.7959802],
    [263.15, 153.0628855],
    [262.15, 148.6300803],
    [261.15, 145.4603510],
    [260.15, 142.0739067],
    [259.15, 139.1362156],
    [258.15, 136.3911599],
    [257.15, 133.6439152],
    [256.15, 130.7784624],
    [255.15, 127.8101247],
    [254.15, 124.9271595],
    [253.15, 122.1842929],
    [252.15, 119.7128672],
    [251.15, 117.5194495],
    [250.15, 115.5033441],
    [249.15, 113.7302220],
    [248.15, 112.3445599],
    [247.15, 111.2478511],
    [246.15, 110.4510407],
    [245.15, 109.9891533],
    [244.15, 109.8490548],
], dtype=float)

t_exp = expdata[:, 0]
y_exp = expdata[:, 1]

x0 = np.array([181.84], dtype=float)

# MATLAB initial guess
k0 = np.array([5.68037900e-08, 8.74400015e+04], dtype=float)


# =========================
# ODE function
# =========================
def kinetic_eqs(t, y, k):
    """
    y: cell volume
    k[0] = a
    k[1] = b
    """
    c = 155.22          # effective membrane area for water transport
    d = 8.314           # gas constant
    e = 5.0             # cooling rate
    f = 18e12           # partial molar volume of water
    g = 109.86          # osmotically inactive cell volume
    h = 2.0             # salt dissociation constant
    ii = 3.33e-13       # intracellular salt molar amount
    jj = 333.88         # latent heat of fusion of ice
    kk = 1e-12          # water density
    l = 273.15          # reference temperature
    m = 4.66e-14        # intracellular CPA moles
    p = 7.103e13        # partial molar volume of CPA

    vol = y[0]

    numerator = vol - g - m * p
    denominator = vol - g - m * p + h * ii / f

    # protect against invalid log/division
    eps = 1e-12
    if numerator <= eps:
        numerator = eps
    if denominator <= eps:
        denominator = eps
    if t <= eps:
        t = eps

    dydt = (
        1e18 * k[0] * c * d * t *
        np.exp(-k[1] / d * (1.0 / t - 1.0 / l)) /
        (e * f) *
        (
            np.log(numerator / denominator)
            - (jj * f * kk / d) * (1.0 / l - 1.0 / t)
        )
    )

    return [dydt]


# =========================
# Simulation
# =========================
def simulate(k, t_eval, x0, method="RK45"):
    sol = solve_ivp(
        fun=lambda t, y: kinetic_eqs(t, y, k),
        t_span=(t_eval[0], t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        method=method,
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t, sol.y.T  # shape: (n_points, 1)


# =========================
# Objective function
# =========================
def objective_function(k, t_exp, y_exp, x0):
    try:
        _, X = simulate(k, t_exp, x0)
        y_model = X[:, 0]

        if np.any(np.isnan(y_model)) or np.any(np.isinf(y_model)):
            return np.ones_like(y_exp) * 1e6

        return y_model - y_exp
    except Exception:
        return np.ones_like(y_exp) * 1e6


# =========================
# Statistics
# =========================
def calc_statistics(y_true, y_pred):
    R = np.corrcoef(y_true, y_pred)[0, 1]
    R2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
    SSE = np.sum((y_true - y_pred) ** 2)

    return {
        "R": R,
        "R2": R2,
        "RMSE": RMSE,
        "SSE": SSE
    }


# =========================
# Main fitting function
# =========================
def fit_model():
    result = least_squares(
        fun=objective_function,
        x0=k0,
        args=(t_exp, y_exp, x0),
        method="trf",
        max_nfev=10000
    )

    k_fit = result.x

    t_fit, X_fit = simulate(k_fit, t_exp, x0)
    y_fit = X_fit[:, 0]

    stats = calc_statistics(y_exp, y_fit)

    print("\nEstimated parameters:")
    print(f"a = {k_fit[0]:.8e}")
    print(f"b = {k_fit[1]:.8e}")

    print("\nFitting statistics:")
    print(f"R    = {stats['R']:.8f}")
    print(f"R^2  = {stats['R2']:.8f}")
    print(f"RMSE = {stats['RMSE']:.8f}")
    print(f"SSE  = {stats['SSE']:.8f}")

    # Plot
    T_model = t_fit - 273.15
    V_model = y_fit / 181.84
    T_exp_c = t_exp - 273.15
    V_exp = y_exp / 181.84

    plt.figure(figsize=(7, 5))
    plt.plot(T_model, V_model, 'b-', label=f"Model simulation (R²={stats['R2']:.4f})")
    plt.plot(T_exp_c, V_exp, 'ro', label="DSC data")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Normalized Cell Volume (V/V0)")
    plt.gca().invert_xaxis()
    plt.ylim([0.5, 1.1])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Export to Excel
    export_df = pd.DataFrame({
        "T_model_C": T_model,
        "V_model_norm": V_model,
        "T_exp_C": T_exp_c,
        "V_exp_norm": V_exp
    })
    export_df.to_excel("abd.xlsx", index=False)

    return result, k_fit, stats, export_df


if __name__ == "__main__":
    fit_model()
