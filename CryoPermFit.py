import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# =========================
# ODE model
# =========================
def kinetic_eqs(t, y, k):
    c = 155.22
    d = 8.314
    e = 5.0
    f = 18e12
    g = 109.86
    h = 2.0
    ii = 3.33e-13
    jj = 333.88
    kk = 1e-12
    l = 273.15
    m = 4.66e-14
    p = 7.103e13

    vol = y[0]

    numerator = vol - g - m * p
    denominator = vol - g - m * p + h * ii / f

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
def simulate(k, t_eval, x0):
    sol = solve_ivp(
        fun=lambda t, y: kinetic_eqs(t, y, k),
        t_span=(t_eval[0], t_eval[-1]),
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T


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
    return R, R2, RMSE, SSE


# =========================
# Parse pasted data
# =========================
def parse_pasted_data(text):
    text = text.strip()
    if not text:
        raise ValueError("No data pasted.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []

    for line in lines:
        if "\t" in line:
            parts = line.split("\t")
        elif "," in line:
            parts = line.split(",")
        else:
            parts = line.split()

        if len(parts) < 2:
            continue

        try:
            t_val = float(parts[0])
            y_val = float(parts[1])
            rows.append([t_val, y_val])
        except ValueError:
            continue

    if len(rows) < 2:
        raise ValueError("Need at least two valid numeric rows.")

    return pd.DataFrame(rows, columns=["Temperature_K", "Volume"])


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CryoPermFit", layout="wide")

st.title("CryoPermFit")
st.write("A tool for estimating osmotic transport parameters of cells at cryogenic conditions.")

st.sidebar.header("Model parameters")

a0 = st.sidebar.number_input("Initial guess: a", value=5.68037900e-08, format="%.8e")
b0 = st.sidebar.number_input("Initial guess: b", value=8.74400015e+04, format="%.8e")
V0 = st.sidebar.number_input("Initial volume x0", value=181.84, format="%.5f")

default_text = """267.99\t181.84
267.15\t178.1667916
266.15\t169.1173019
265.15\t165.6389179
264.15\t158.7959802
263.15\t153.0628855
262.15\t148.6300803
261.15\t145.460351
260.15\t142.0739067
259.15\t139.1362156
258.15\t136.3911599
257.15\t133.6439152
256.15\t130.7784624
255.15\t127.8101247
254.15\t124.9271595
253.15\t122.1842929
252.15\t119.7128672
251.15\t117.5194495
250.15\t115.5033441
249.15\t113.730222
248.15\t112.3445599
247.15\t111.2478511
246.15\t110.4510407
245.15\t109.9891533
244.15\t109.8490548"""

st.subheader("Paste experimental data")
st.caption("Paste two columns: Temperature(K) and Volume. Supports tab, comma, or space separated values.")

data_text = st.text_area("Data", value=default_text, height=320)

run_fit = st.button("Start Fitting")

if run_fit:
    try:
        data = parse_pasted_data(data_text)
        t_exp = data["Temperature_K"].values
        y_exp = data["Volume"].values
        x0 = np.array([V0], dtype=float)
        k0 = np.array([a0, b0], dtype=float)

        with st.spinner("Running fitting..."):
            result = least_squares(
                fun=objective_function,
                x0=k0,
                args=(t_exp, y_exp, x0),
                method="trf",
                max_nfev=10000
            )

            k_fit = result.x
            _, X_fit = simulate(k_fit, t_exp, x0)
            y_fit = X_fit[:, 0]

            R, R2, RMSE, SSE = calc_statistics(y_exp, y_fit)

        st.success("Fitting completed.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fitted parameters")
            st.dataframe(pd.DataFrame({
                "Parameter": ["a", "b"],
                "Value": [k_fit[0], k_fit[1]]
            }))

        with col2:
            st.subheader("Statistics")
            st.dataframe(pd.DataFrame({
                "Metric": ["R", "R²", "RMSE", "SSE"],
                "Value": [R, R2, RMSE, SSE]
            }))

        T_model = t_exp - 273.15
        V_model = y_fit / V0
        T_exp_c = t_exp - 273.15
        V_exp = y_exp / V0

# 页面设置（放最上面）
st.set_page_config(layout="wide")


# ===== 图1 =====
fig1, ax1 = plt.subplots(figsize=(5, 3))
ax1.plot(T_model, V_model, 'b-', label=f"Model (R²={R2:.4f})")
ax1.plot(T_exp_c, V_exp, 'ro', label="Data")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Normalized Volume")
ax1.invert_xaxis()
ax1.legend()


# ===== 图2 =====
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(T_exp_c, residual, 'ko-')
ax2.axhline(0, linestyle="--")
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Residual")
ax2.invert_xaxis()


# ===== 并排显示 =====
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.pyplot(fig2, use_container_width=True)

        export_df = pd.DataFrame({
            "T_model_C": T_model,
            "V_model_norm": V_model,
            "T_exp_C": T_exp_c,
            "V_exp_norm": V_exp,
            "Residual": residual
        })

        st.subheader("Fitted data")
        st.dataframe(export_df)

        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download fitted data as CSV",
            data=csv_data,
            file_name="CryoPermFit_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
