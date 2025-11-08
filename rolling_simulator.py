import streamlit as st
import math
import plotly.graph_objects as go
import numpy as np
from itertools import product

import pandas as pd
from fpdf import FPDF
import datetime
import os
import unicodedata



# ==============================
# App Identity & Page Setup
# ==============================
APP_NAME = "MILLOPT"
TAGLINE = "Process-Level Rolling Simulation & Optimization Suite"

st.set_page_config(page_title=f"{APP_NAME} – Rolling Simulator", layout="wide")

# ==============================
# THEME: Dark Industrial CSS
# ==============================
st.markdown("""
<style>
/* Base */
html, body, [class*="css"]  {
  background-color: #0d1117 !important;
  color: #e5e7eb !important;
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
}

/* Container padding */
div.block-container { padding-top: 0.75rem; padding-bottom: 0.5rem; }

/* Header bar */
.m-header {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(90deg, #0f141c 0%, #0d1117 100%);
  border: 1px solid #1f2937; border-radius: 12px;
  padding: 12px 16px; margin-bottom: 12px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25), inset 0 0 0 1px rgba(255,255,255,0.02);
}
.m-left { display:flex; align-items:center; gap: 12px;}
.m-logo {
  width: 38px; height: 38px; border-radius: 8px;
  background: conic-gradient(from 180deg at 50% 50%, #00b4d8, #1f77b4, #0ea5e9, #00b4d8);
  box-shadow: 0 0 18px rgba(31,119,180,0.35);
}
.m-title { line-height:1.1; }
.m-title h1 {
  color: #e6f6ff; font-size: 1.35rem; font-weight: 700; margin: 0;
  letter-spacing: 0.2px;
}
.m-title span {
  color: #9fb9c8; font-size: 0.85rem; font-weight: 500;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #0f141c !important;
  border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #cbe8f6 !important;
}
.sidebar-card {
  background: #121923; border: 1px solid #1f2937; border-radius: 12px; padding: 10px 12px; margin-bottom: 10px;
}

/* KPI Cards */
.kpi-wrap { display:flex; gap: 12px; }
.kpi {
  flex:1; background: radial-gradient(1200px 400px at -10% -10%, rgba(31,119,180,0.15), rgba(0,0,0,0)) , #11161f;
  border: 1px solid #1f2937; border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.25), inset 0 0 0 1px rgba(255,255,255,0.02);
}
.kpi h3 { margin: 0; color: #9fd6ff; font-size: 0.9rem; font-weight: 600; letter-spacing: .3px;}
.kpi .val { margin-top: 6px; color: #ffffff; font-size: 1.65rem; font-weight: 700; }
.kpi .sub { margin-top: 2px; color: #9fb9c8; font-size: 0.8rem; }

/* Tabs -> like suite modules */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background-color: #11161f; color: #cbd5e1; border-radius: 10px; padding: 8px 14px; border: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(180deg, #00b4d8 0%, #1f77b4 100%); color: #ffffff; border: 1px solid #2aa8e0;
}

/* Plot container */
.stPlotlyChart { background-color: #0d1117 !important; border-radius: 12px; }

/* Section headings */
h2, h3 { color: #9fd6ff !important; }

/* Footer */
.m-footer {
  border-top: 1px solid #1f2937; margin-top: 18px; padding-top: 10px; color: #8aa0ae; text-align: center;
  font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown(f"""
<div class="m-header">
  <div class="m-left">
    <div class="m-logo"></div>
    <div class="m-title">
      <h1>{APP_NAME}</h1>
      <span>{TAGLINE}</span>
    </div>
  </div>
  <div class="m-right" style="opacity:.9;color:#9fb9c8;font-size:.85rem;">
    v1.0 · Dark Industrial Skin
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Material & Steel Databases
# ==============================
ROLLS = {
    "SG Iron": {"E": 170e9, "nu": 0.28, "k_roll": 40.0, "HB": 200, "k_range": (1.0, 1.2)},
    "DPIC":    {"E": 190e9, "nu": 0.28, "k_roll": 32.0, "HB": 450, "k_range": (1.2, 1.5)}
}
STEELS = {
    "Low-C Steel": {"A": 1.0e13, "n": 5.0, "alpha": 0.012, "Q": 320e3},
    "IF Steel":    {"A": 5.0e12, "n": 4.8, "alpha": 0.011, "Q": 300e3}
}
R_GAS = 8.314  # J/mol-K

# ==============================
# Sidebar (Inputs)
# ==============================
st.sidebar.header("🔧 Inputs")

with st.sidebar.container():
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### Roll Material")
    material_choice = st.radio(
        "",
        list(ROLLS.keys()),
        horizontal=True,
        key="roll_mat"
    )

    st.markdown("#### Steel Grade (Flow Stress Model)")
    steel_choice = st.radio(
        "",
        list(STEELS.keys()),
        horizontal=True,
        key="steel_grade"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    h1 = st.number_input("Initial thickness h₁ [mm]", 10.0, 50.0, 30.0)
    h2 = st.number_input("Final thickness h₂ [mm]", 1.0, h1 - 0.5, 10.0)
    D  = st.number_input("Roll diameter D [mm]", 300.0, 1000.0, 500.0)
    b  = st.number_input("Strip width b [mm]", 500.0, 2000.0, 1000.0)
    N_rpm = st.number_input("Roll speed N [rpm]", 50.0, 300.0, 100.0)
    T_C = st.slider("Rolling Temperature [°C]", 700, 1200, 950, step=10)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    mu0 = st.slider("Base interface friction μ₀", 0.05, 0.30, 0.12, step=0.01)
    a_T = st.slider("Temp sensitivity a (Δμ/°C)", 0.0, 0.0005, 0.0001, step=0.00005)
    b_v = st.slider("Speed sensitivity b (Δμ per ln v)", 0.0, 0.05, 0.01, step=0.005)
    st.markdown('</div>', unsafe_allow_html=True)

roll = ROLLS[material_choice]
steel = STEELS[steel_choice]

# ==============================
# Helper Models (UNCHANGED LOGIC)
# ==============================
def surface_speed(D_mm, N_rpm):
    D_m = D_mm / 1000
    return math.pi * D_m * (N_rpm / 60.0)  # m/s

def strain_rate_estimate(h1_mm, h2_mm, D_mm, N_rpm):
    v = surface_speed(D_mm, N_rpm)
    R = (D_mm / 2) / 1000
    dh = (h1_mm - h2_mm) / 1000
    L = max(1e-8, math.sqrt(max(1e-12, R * max(1e-9, dh))))
    return max(1e-6, v / L / math.sqrt(3))

def flow_stress_ZH(T_C, edot, A, n, alpha, Q):
    T_K = T_C + 273.15
    Z = edot * math.exp(Q / (R_GAS * T_K))
    sigma = (1.0 / alpha) * math.asinh(max(1e-12, (Z / A)) ** (1.0 / n)) / 1e6
    return max(1.0, sigma)

def mu_of_T_v(mu0, a, b_sens, T_C, v, k_roll):
    roll_penalty = (40.0 / max(10.0, k_roll))
    mu = mu0 + a * (T_C - 900.0) - b_sens * math.log(max(v, 1e-3))
    mu *= roll_penalty
    return float(np.clip(mu, 0.03, 0.35))

def friction_factor_from_mu(mu):
    return float((2.0 / math.pi) * math.atan((math.pi * mu) / 2.0))

def effective_radius(R_m, F_N, b_m, E, nu):
    Eprime = E / (1.0 - nu**2)
    dR = F_N / (math.pi * max(1e-6, b_m) * max(1e9, Eprime))
    return R_m + dR

def simple_mean_pressure(sigma_bar, m):
    return sigma_bar * (1.0 + m)

def contact_length(R_m, h1_m, h2_m):
    return math.sqrt(max(1e-12, R_m * max(1e-9, (h1_m - h2_m))))

# ==============================
# Core Calculation (UNCHANGED)
# ==============================
def compute_F_T_P(h1, h2, D, b, N_rpm, T_C, steel, roll, mu0, a_T, b_v):
    R0 = (D / 2) / 1000
    h1_m, h2_m, b_m = h1 / 1000, h2 / 1000, b / 1000

    v = surface_speed(D, N_rpm)
    edot = strain_rate_estimate(h1, h2, D, N_rpm)
    sigma_bar = flow_stress_ZH(T_C, edot, steel["A"], steel["n"], steel["alpha"], steel["Q"])
    mu = mu_of_T_v(mu0, a_T, b_v, T_C, v, roll["k_roll"])
    m = friction_factor_from_mu(mu)

    R_eff = R0
    F = 0.0
    for _ in range(2):
        L = contact_length(R_eff, h1_m, h2_m)
        p_avg = simple_mean_pressure(sigma_bar, m) * 1e6
        F = p_avg * L * b_m
        R_eff = effective_radius(R0, F, b_m, roll["E"], roll["nu"])

    Tq = (F * L) / 2.0
    N_rps = N_rpm / 60.0
    P = 2.0 * math.pi * N_rps * Tq

    return {"F": F, "Tq": Tq, "P": P, "sigma_bar": sigma_bar, "mu": mu, "m": m, "R_eff": R_eff, "L": L, "v": v, "edot": edot}

# ==============================
# Current Calculation & KPI Cards
# ==============================
res = compute_F_T_P(h1, h2, D, b, N_rpm, T_C, steel, roll, mu0, a_T, b_v)

st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Rolling Load</h3>
    <div class="val">{res['F']/1000:.2f} kN</div>
    <div class="sub">Contact Length: {res['L']*1000:.1f} mm</div>
  </div>
""", unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Torque</h3>
    <div class="val">{res['Tq']/1000:.2f} kN·m</div>
    <div class="sub">Surface Speed: {res['v']:.2f} m/s</div>
  </div>
""", unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Power</h3>
    <div class="val">{res['P']/1000:.2f} kW</div>
    <div class="sub">Flow Stress σ̄: {res['sigma_bar']:.1f} MPa</div>
  </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.caption(f"μ={res['mu']:.3f} (m={res['m']:.3f}) · ε̇={res['edot']:.2f} s⁻¹ · R_eff≈{(D/2)/1000:.4f}→{res['R_eff']:.4f} m")

# ==============================
# Suite Tabs (same functionality)
# ==============================
TAB1, TAB2, TAB3 = st.tabs(["📊 Dashboard", "🆚 Compare", "🧠 Optimize"])

with TAB1:
    st.subheader("Rolling Behavior & Sensitivity")

    # Load vs Reduction Ratio @ multiple temps
    T_list = [800, 950, 1100]
    ratios = np.linspace(5, 50, 100)
    fig1 = go.Figure()
    for Tplot in T_list:
        loads = []
        for r in ratios:
            h2_temp = h1 - (h1 * r / 100)
            tmp = compute_F_T_P(h1, h2_temp, D, b, N_rpm, Tplot, steel, roll, mu0, a_T, b_v)
            loads.append(tmp["F"] / 1000)
        fig1.add_trace(go.Scatter(x=ratios, y=loads, mode="lines", name=f"{Tplot}°C"))
    fig1.update_layout(
        title=f"Load vs Reduction Ratio ({material_choice})",
        xaxis_title="Reduction Ratio (%)", yaxis_title="Load (kN)",
        template="plotly_dark", hovermode="x unified"
    )
    st.plotly_chart(fig1, width='stretch')

    # Torque vs Speed
    speeds = np.linspace(50, 300, 80)
    torq_vs_speed = []
    for n in speeds:
        tmp = compute_F_T_P(h1, h2, D, b, n, T_C, steel, roll, mu0, a_T, b_v)
        torq_vs_speed.append(tmp["Tq"]/1000)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=speeds, y=torq_vs_speed, mode="lines", name="Torque vs Speed"))
    fig2.update_layout(title="Torque vs Roll Speed", xaxis_title="Speed (rpm)", yaxis_title="Torque (kN·m)",
                       template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig2, width='stretch')

    # Torque vs Temperature
    temps = np.linspace(700, 1200, 50)
    torq_vs_temp = []
    for Tplot in temps:
        tmp = compute_F_T_P(h1, h2, D, b, N_rpm, Tplot, steel, roll, mu0, a_T, b_v)
        torq_vs_temp.append(tmp["Tq"]/1000)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=temps, y=torq_vs_temp, mode="lines", name="Torque vs Temp"))
    fig3.update_layout(title="Torque vs Temperature", xaxis_title="Temperature (°C)", yaxis_title="Torque (kN·m)",
                       template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig3, width='stretch')

    # Power vs Friction Proxy (k)
    k_vals = np.linspace(ROLLS[material_choice]["k_range"][0], ROLLS[material_choice]["k_range"][1], 60)
    P_vs_k = []
    for kf in k_vals:
        mu_equiv = float(np.clip(kf - 0.9, 0.03, 0.35))
        v = surface_speed(D, N_rpm)
        edot = strain_rate_estimate(h1, h2, D, N_rpm)
        sigma_bar = flow_stress_ZH(T_C, edot, steel["A"], steel["n"], steel["alpha"], steel["Q"])
        m = friction_factor_from_mu(mu_equiv)
        R0 = (D/2)/1000; h1_m, h2_m, b_m = h1/1000, h2/1000, b/1000
        R_eff = R0
        for _ in range(2):
            L = contact_length(R_eff, h1_m, h2_m)
            p_avg = simple_mean_pressure(sigma_bar, m) * 1e6
            F_k = p_avg * L * b_m
            R_eff = effective_radius(R0, F_k, b_m, roll["E"], roll["nu"])
        Tq_k = (F_k * L) / 2.0
        P_k = 2.0 * math.pi * (N_rpm/60.0) * Tq_k
        P_vs_k.append(P_k/1000)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=k_vals, y=P_vs_k, mode="lines"))
    fig4.update_layout(title="Power vs Friction Factor (proxy)", xaxis_title="Friction Factor k (proxy)",
                       yaxis_title="Power (kW)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig4, width='stretch')

with TAB2:
    st.subheader("🆚 SG Iron vs DPIC — Comparative Performance Analysis")

    # Base settings
    mats = ["SG Iron", "DPIC"]
    ratios = np.linspace(5, 50, 60)
    temp_range = np.linspace(800, 1150, 8)  # Temperature sweep for comparison

    # -------------------------------
    # 1️⃣ Load vs Reduction Ratio (existing comparison)
    # -------------------------------
    st.markdown("### 1️⃣ Load vs Reduction Ratio (at selected temperature)")
    figc1 = go.Figure()
    for mname in mats:
        rroll = ROLLS[mname]
        loads = []
        for r in ratios:
            h2_temp = h1 - (h1 * r / 100)
            tmp = compute_F_T_P(h1, h2_temp, D, b, N_rpm, T_C, steel, rroll, mu0, a_T, b_v)
            loads.append(tmp["F"]/1000)
        figc1.add_trace(go.Scatter(x=ratios, y=loads, mode="lines", name=mname))
    figc1.update_layout(
        title=f"Load vs Reduction Ratio @ {T_C}°C",
        xaxis_title="Reduction Ratio (%)",
        yaxis_title="Rolling Load (kN)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(figc1, width='stretch')

    # -------------------------------
    # 2️⃣ Load vs Temperature
    # -------------------------------
    st.markdown("### 2️⃣ Load vs Temperature")
    figc2 = go.Figure()
    for mname in mats:
        rroll = ROLLS[mname]
        loads_T = []
        for Tplot in temp_range:
            tmp = compute_F_T_P(h1, h2, D, b, N_rpm, Tplot, steel, rroll, mu0, a_T, b_v)
            loads_T.append(tmp["F"]/1000)
        figc2.add_trace(go.Scatter(x=temp_range, y=loads_T, mode="lines+markers", name=mname))
    figc2.update_layout(
        title="Load vs Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="Load (kN)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(figc2, width='stretch')

    # -------------------------------
    # 3️⃣ Power vs Temperature
    # -------------------------------
    st.markdown("### 3️⃣ Power vs Temperature")
    figc3 = go.Figure()
    for mname in mats:
        rroll = ROLLS[mname]
        power_T = []
        for Tplot in temp_range:
            tmp = compute_F_T_P(h1, h2, D, b, N_rpm, Tplot, steel, rroll, mu0, a_T, b_v)
            power_T.append(tmp["P"]/1000)
        figc3.add_trace(go.Scatter(x=temp_range, y=power_T, mode="lines+markers", name=mname))
    figc3.update_layout(
        title="Power vs Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="Power (kW)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(figc3, width='stretch')

    # -------------------------------
    # 4️⃣ Energy Efficiency vs Temperature
    # -------------------------------
    st.markdown("### 4️⃣ Energy Consumption vs Temperature")
    figc4 = go.Figure()
    for mname in mats:
        rroll = ROLLS[mname]
        energy_T = []
        for Tplot in temp_range:
            tmp = compute_F_T_P(h1, h2, D, b, N_rpm, Tplot, steel, rroll, mu0, a_T, b_v)
            b_m, h2_m = b/1000, h2/1000
            density_steel = 7850
            mass_flow_tph = density_steel * b_m * h2_m * tmp["v"] * 3600 / 1000
            energy_kWh_ton = tmp["P"]/max(mass_flow_tph, 1e-6)
            energy_T.append(energy_kWh_ton)
        figc4.add_trace(go.Scatter(x=temp_range, y=energy_T, mode="lines+markers", name=mname))
    figc4.update_layout(
        title="Energy Consumption vs Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="Energy (kWh/ton)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(figc4, width='stretch')

    # -------------------------------
    # 5️⃣ Summary Table at Current T
    # -------------------------------
    st.markdown("### 5️⃣ Summary Comparison at Current Temperature")
    F_vals, P_vals, E_vals = [], [], []
    for mname in mats:
        tmp = compute_F_T_P(h1, h2, D, b, N_rpm, T_C, steel, ROLLS[mname], mu0, a_T, b_v)
        b_m, h2_m = b/1000, h2/1000
        density_steel = 7850
        mass_flow_tph = density_steel * b_m * h2_m * tmp["v"] * 3600 / 1000
        energy_kWh_ton = tmp["P"]/max(mass_flow_tph, 1e-6)
        F_vals.append(tmp["F"]/1000)
        P_vals.append(tmp["P"]/1000)
        E_vals.append(energy_kWh_ton)

    figb = go.Figure(data=[
        go.Bar(name='Load (kN)', x=mats, y=F_vals),
        go.Bar(name='Power (kW)', x=mats, y=P_vals),
        go.Bar(name='Energy (kWh/ton)', x=mats, y=E_vals)
    ])
    figb.update_layout(
        barmode='group',
        title=f"Material Comparison Summary @ {T_C}°C",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(figb, width='stretch')

    st.caption("Graphs show comparative performance of SG Iron vs DPIC rolls across temperature range.")


with TAB3:
    st.subheader("Optimization – Find Minimum Power / Best Efficiency")
    st.caption("Grid-search optimization over Temperature, Speed, and Reduction (h₂).")

    cA, cB, cC = st.columns(3)
    T_min = cA.number_input("T min [°C]", 750, 1100, 850)
    T_max = cA.number_input("T max [°C]", 800, 1200, 1050)
    N_min = cB.number_input("N min [rpm]", 50, 300, 80)
    N_max = cB.number_input("N max [rpm]", 60, 350, 180)
    r_min = cC.slider("Reduction min [%]", 5, 40, 10)
    r_max = cC.slider("Reduction max [%]", 10, 50, 30)
    grid_density = st.select_slider("Grid density", options=["coarse","medium","fine"], value="medium")

    if grid_density == "coarse":
        nT, nN, nR = 6, 6, 6
    elif grid_density == "fine":
        nT, nN, nR = 16, 16, 16
    else:
        nT, nN, nR = 10, 10, 10

    if st.button("Run Optimization", type="primary"):
        T_vals = np.linspace(T_min, T_max, nT)
        N_vals = np.linspace(N_min, N_max, nN)
        R_vals = np.linspace(r_min, r_max, nR)

        best = None
        pts_x, pts_y, pts_color = [], [], []

        for Topt, Nopt, ropt in product(T_vals, N_vals, R_vals):
            h2_opt = h1 - (h1 * ropt / 100.0)
            out = compute_F_T_P(h1, h2_opt, D, b, Nopt, Topt, steel, roll, mu0, a_T, b_v)
            PkW = out["P"]/1000
            eff = PkW / max(1e-6, (h1 - h2_opt))
            if (best is None) or (PkW < best[0]):
                best = (PkW, Topt, Nopt, ropt, eff, out)
            pts_x.append(ropt); pts_y.append(PkW); pts_color.append(Topt)

        st.success(f"Min Power: {best[0]:.2f} kW at T={best[1]:.0f}°C, N={best[2]:.0f} rpm, Reduction={best[3]:.1f}%")
        st.caption(f"Efficiency proxy (kW per mm reduction): {best[4]:.4f}")

        figopt = go.Figure()
        figopt.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers",
            marker=dict(size=6, color=pts_color, colorscale='Viridis'),
            name="Grid Points"))
        figopt.add_trace(go.Scatter(
            x=[best[3]], y=[best[0]], mode="markers",
            marker=dict(size=12, color='red'), name="Optimum"))
        figopt.update_layout(
            title="Optimization Cloud: Power vs Reduction (colored by Temperature)",
            xaxis_title="Reduction (%)", yaxis_title="Power (kW)",
            template="plotly_dark", hovermode="x unified"
        )
        st.plotly_chart(figopt, width='stretch')

        with st.expander("Details at Optimum"):
            out = best[5]
            st.write({
                "σ̄ (MPa)": round(out['sigma_bar'], 2),
                "μ": round(out['mu'], 3),
                "m": round(out['m'], 3),
                "L (mm)": round(out['L']*1000, 2),
                "R_eff (mm)": round(out['R_eff']*1000, 3),
                "v (m/s)": round(out['v'], 3),
            })



# ==============================
# ENERGY & CO₂ MODULE (Corrected)
# ==============================
density_steel = 7850  # kg/m³
b_m, h2_m = b / 1000, h2 / 1000

# Throughput estimation (mass flow in tons/hour)
mass_flow_tph = density_steel * b_m * h2_m * res["v"] * 3600 / 1000  # ton/hour

if mass_flow_tph > 1e-6:
    energy_kWh_ton = res["P"] / max(mass_flow_tph, 1e-6)  # kW per ton/hour = kWh/ton
else:
    energy_kWh_ton = 0.0

# CO₂ calculation (typical Indian grid factor)
co2_factor = 0.6  # kg CO₂ per kWh
co2_kg_per_ton = energy_kWh_ton * co2_factor

st.markdown("---")
st.subheader("♻️ Energy & Environmental Analysis")
colE1, colE2 = st.columns(2)
colE1.metric("Energy Consumption", f"{energy_kWh_ton:.1f} kWh/ton")
colE2.metric("CO₂ Emission", f"{co2_kg_per_ton:.1f} kg CO₂/ton")
st.caption("Calculated using instantaneous process power and strip mass throughput (tons/hour).")


# ==============================
# LOGGING MODULE
# ==============================
LOG_FILE = "roll_log.csv"

def log_run():
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roll_material": material_choice,
        "steel": steel_choice,
        "T_C": T_C,
        "N_rpm": N_rpm,
        "h1": h1,
        "h2": h2,
        "D": D,
        "b": b,
        "Load_kN": res["F"]/1000,
        "Torque_kNm": res["Tq"]/1000,
        "Power_kW": res["P"]/1000,
        "Energy_kWh_ton": energy_kWh_ton,
        "CO2_kg_ton": co2_kg_per_ton
    }
    df = pd.DataFrame([data])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
    st.success("✅ Simulation logged successfully!")

if st.button("🧾 Log This Simulation"):
    log_run()

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        st.download_button("📥 Download Log File (CSV)", data=f, file_name="roll_log.csv", mime="text/csv")


# ==============================
# UNICODE-SAFE TEXT SANITIZER
# ==============================
def clean_text(text):
    """Make text Latin-1 safe for FPDF (replaces subscripts, degree, etc.)"""
    replacements = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "·": ".", "°": " deg", "²": "^2", "³": "^3", "—": "-", "–": "-"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text

# ==============================
# REPORT GENERATOR (FPDF)
# ==============================
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, clean_text(f"{APP_NAME} - Rolling Simulation Report"), ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, datetime.datetime.now().strftime("Generated on %Y-%m-%d %H:%M:%S"), ln=True, align="C")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, clean_text(title), ln=True)
        self.ln(2)

    def add_line(self, label, value):
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, clean_text(f"{label}: {value}"), ln=True)


if st.button("📄 Generate PDF Report"):
    pdf = PDF()
    pdf.add_page()
    pdf.section_title("Input Parameters")
    inputs = {
        "Roll Material": material_choice,
        "Steel Grade": steel_choice,
        "Temperature (°C)": T_C,
        "Speed (rpm)": N_rpm,
        "Thickness (mm)": f"h1={h1}, h2={h2}",
        "Roll Dia (mm)": D,
        "Width (mm)": b
    }
    for k, v in inputs.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    pdf.section_title("Simulation Results")
    results = {
        "Load (kN)": f"{res['F']/1000:.2f}",
        "Torque (kN·m)": f"{res['Tq']/1000:.2f}",
        "Power (kW)": f"{res['P']/1000:.2f}",
        "Energy (kWh/ton)": f"{energy_kWh_ton:.2f}",
        "CO2 (kg/ton)": f"{co2_kg_per_ton:.2f}"
    }
    for k, v in results.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    pdf.section_title("Notes")
    pdf.multi_cell(0, 6, clean_text(
        "This report summarizes the process simulation run performed in MILLOPT.\n"
        "Energy and CO2 values are estimates assuming steady-state throughput.\n"
        "Results are indicative for comparative and optimization purposes."
    ))

    filename = f"MILLOPT_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    with open(filename, "rb") as f:
        st.download_button("📤 Download Report (PDF)", f, file_name=filename, mime="application/pdf")


# ==============================
# FOOTER
# ==============================
st.markdown(f"""
<div class="m-footer">
  © 2025 {APP_NAME} · All rights reserved · Built for industrial rolling analysis
</div>
""", unsafe_allow_html=True)

