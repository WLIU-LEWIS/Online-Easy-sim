import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from io import StringIO


# =========================
# ODE model
# =========================
def kinetic_eqs(t, x, k):
    a = 513.098453785684
    r = 0.082057338 * 10**15
    tt = 283.15
    mne = 0.3 * 10**(-15)
    mn0 = 0.3 * 10**(-15)
    v0 = 590.05351792762
    vb = 385.487547
    vs0 = 0
    sigma = 0.6
    mse = 2 * 10**(-15)
    vsm = 0.071 * 10**15

    V = x[0]
    Nsi = x[1]

    vs = vsm * Nsi

    denom1 = V - vb - vs
    denom2 = V - vb - vs - 0.0166 * 10**15 * 0.3 * 10**(-15) * v0

    if abs(denom1) < 1e-12:
        denom1 = np.sign(denom1 + 1e-12) * 1e-12
    if abs(denom2) < 1e-12:
        denom2 = np.sign(denom2 + 1e-12) * 1e-12

    mni = mn0 * (v0 - vb - vs0) / denom1

    dVdt = (
        k[0] * a * r * tt *
        (mni - mne + sigma * (Nsi / denom2 - mse))
        + vb + vs + 0.0166 * 10**15 * 0.3 * 10**(-15) * v0
    )

    dNsidt = k[1] * a * (mse - Nsi / denom2)

    return [dVdt, dNsidt]


# =========================
# Simulation
# =========================
def simulate(k, t_eval, x0):
    sol = solve_ivp(
        lambda t, x: kinetic_eqs(t, x, k),
        (t_eval[0], t_eval[-1]),
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y.T


# =========================
# Objective function
# =========================
def objective(k, t, y, x0):
    try:
        X = simulate(k, t, x0)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return np.ones_like(y) * 1e6
        return X[:, 0] - y
    except Exception:
        return np.ones_like(y) * 1e6


# =========================
# Parse pasted text
# =========================
def parse_pasted_data(text):
    text = text.strip()
    if not text:
        raise ValueError("No data pasted.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []

    for line in lines:
        # 支持 tab、逗号、空格
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
            # 跳过表头或非数字行
            continue

    if len(rows) < 2:
        raise ValueError("Need at least two valid rows of numeric data.")

    df = pd.DataFrame(rows, columns=["time", "V"])
    return df


# =========================
# UI
# =========================
st.title("Cell Volume Model Fitting Tool")

st.sidebar.header("Input parameters")

Lp0 = st.sidebar.number_input("Initial Lp", value=0.5)
Ps0 = st.sidebar.number_input("Initial Ps", value=5.0)

Lp_lb = st.sidebar.number_input("Lp lower bound", value=0.001)
Ps_lb = st.sidebar.number_input("Ps lower bound", value=0.1)

Lp_ub = st.sidebar.number_input("Lp upper bound", value=1.0)
Ps_ub = st.sidebar.number_input("Ps upper bound", value=20.0)

V0 = st.sidebar.number_input("Initial V", value=590.0535179)
Nsi0 = st.sidebar.number_input("Initial Nsi", value=0.0)

st.subheader("Paste data directly")
st.caption("Paste two-column data: time and V. Supports tab, comma, or space separated values.")

default_text = """0\t590.0535179
2\t751.0950383
4\t863.5021225
8\t939.7530357
10\t960.5514376
12\t1018.161594
14\t1041.420706
16\t1058.237245
18\t1026.750751
20\t1059.661608
22\t1054.89364
24\t1046.978016
26\t1046.397222
28\t1051.6396
30\t1027.570455
32\t1030.693258
34\t1020.610731
36\t1000.885289
38\t1042.496781
40\t1011.5685"""

data_text = st.text_area(
    "Paste your data here",
    value=default_text,
    height=300
)

start_fit = st.button("Start Fitting")

if start_fit:
    try:
        data = parse_pasted_data(data_text)

        t = data["time"].values
        y = data["V"].values

        if len(t) < 2:
            st.error("Please provide at least two rows of valid data.")
            st.stop()

        if not np.all(np.diff(t) >= 0):
            st.error("Time values must be in nondecreasing order.")
            st.stop()

        st.write("### Parsed data")
        st.dataframe(data)

        k0 = np.array([Lp0, Ps0], dtype=float)
        lb = np.array([Lp_lb, Ps_lb], dtype=float)
        ub = np.array([Lp_ub, Ps_ub], dtype=float)
        x0 = np.array([V0, Nsi0], dtype=float)

        if np.any(k0 < lb) or np.any(k0 > ub):
            st.error("Initial parameter guesses must be within the bounds.")
            st.stop()

        with st.spinner("Running fitting..."):
            result = least_squares(
                objective,
                k0,
                bounds=(lb, ub),
                args=(t, y, x0),
                method="trf",
                max_nfev=5000
            )

            k = result.x
            X = simulate(k, t, x0)
            y_fit = X[:, 0]

            R = np.corrcoef(y, y_fit)[0, 1]
            R2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)
            RMSE = np.sqrt(np.mean((y - y_fit) ** 2))
            SSE = np.sum((y - y_fit) ** 2)

        st.success("Fitting completed.")

        st.write("### Fitted parameters")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Lp", "Ps"],
            "Value": [k[0], k[1]]
        }))

        st.write("### Statistics")
        st.dataframe(pd.DataFrame({
            "Metric": ["R", "R²", "RMSE", "SSE"],
            "Value": [R, R2, RMSE, SSE]
        }))

        t_smooth = np.linspace(t.min(), t.max(), 300)
        y_smooth = simulate(k, t_smooth, x0)[:, 0]

        fig1, ax1 = plt.subplots()
        ax1.plot(t_smooth, y_smooth, label="Model fit")
        ax1.scatter(t, y, label="Experimental data")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("V")
        ax1.set_title("Fitting result")
        ax1.legend()
        st.pyplot(fig1)

        residual = y - y_fit
        fig2, ax2 = plt.subplots()
        ax2.plot(t, residual, "o-")
        ax2.axhline(0, linestyle="--")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Residual")
        ax2.set_title("Residual plot")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
