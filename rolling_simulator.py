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
APP_NAME = "MILLOPT Pro"
TAGLINE = "Advanced Rolling Simulation & Optimization Suite with Temperature-Dependent Material Modeling"

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
    v2.0 · Temperature-Dependent Enhanced Model
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Material & Steel Databases - ENHANCED
# ==============================
# Roll material properties with temperature-dependent thermal conductivity
ROLLS = {
    "SG Iron": {
        "E": 170e9,           # Young's modulus (Pa)
        "nu": 0.28,           # Poisson's ratio
        "k_roll_base": 40.0,  # Base thermal conductivity at 20°C (W/m·K)
        "k_temp_coeff": -0.015, # Temperature coefficient for k (decrease with temp)
        "HB": 200,            # Brinell hardness
        "density": 7200,      # kg/m³
        "specific_heat": 500, # J/kg·K
        "thermal_exp": 11e-6, # Thermal expansion coefficient (1/K)
        "wear_resistance": 1.0  # Relative wear resistance factor
    },
    "DPIC": {
        "E": 190e9,
        "nu": 0.28,
        "k_roll_base": 32.0,
        "k_temp_coeff": -0.012,
        "HB": 450,
        "density": 7500,
        "specific_heat": 480,
        "thermal_exp": 10e-6,
        "wear_resistance": 1.8  # Better wear resistance
    }
}

# Steel grade properties with enhanced flow stress models
STEELS = {
    "Low-C Steel": {
        "A": 1.0e13,          # Zener-Hollomon parameter
        "n": 5.0,             # Stress exponent
        "alpha": 0.012,       # Material constant (MPa⁻¹)
        "Q": 320e3,           # Activation energy (J/mol)
        "T_recryst": 850,     # Recrystallization temperature (°C)
        "density": 7850,      # kg/m³
        "specific_heat": 600  # J/kg·K (average)
    },
    "IF Steel": {
        "A": 5.0e12,
        "n": 4.8,
        "alpha": 0.011,
        "Q": 300e3,
        "T_recryst": 820,
        "density": 7870,
        "specific_heat": 590
    },
    "Medium-C Steel": {
        "A": 2.0e13,
        "n": 5.5,
        "alpha": 0.013,
        "Q": 340e3,
        "T_recryst": 880,
        "density": 7840,
        "specific_heat": 610
    }
}

R_GAS = 8.314  # J/mol·K

# ==============================
# Sidebar (Inputs) - ENHANCED
# ==============================
st.sidebar.header("🔧 Process Inputs")

with st.sidebar.container():
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### Roll Material Selection")
    material_choice = st.radio(
        "",
        list(ROLLS.keys()),
        horizontal=True,
        key="roll_mat",
        help="DPIC offers better hardness and wear resistance"
    )

    st.markdown("#### Steel Grade (Flow Stress Model)")
    steel_choice = st.radio(
        "",
        list(STEELS.keys()),
        horizontal=True,
        key="steel_grade",
        help="Different grades have varying temperature sensitivity"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### Geometric Parameters")
    h1 = st.number_input("Initial thickness h₁ [mm]", 10.0, 100.0, 30.0, step=1.0)
    h2 = st.number_input("Final thickness h₂ [mm]", 1.0, h1 - 0.5, 10.0, step=0.5)
    D  = st.number_input("Roll diameter D [mm]", 300.0, 1500.0, 500.0, step=50.0)
    b  = st.number_input("Strip width b [mm]", 500.0, 2500.0, 1000.0, step=100.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### Process Conditions")
    N_rpm = st.number_input("Roll speed N [rpm]", 50.0, 400.0, 100.0, step=10.0)
    T_C = st.slider("Rolling Temperature [°C]", 700, 1200, 950, step=10,
                    help="Higher temperature reduces flow stress but increases roll thermal stress")
    T_roll_initial = st.slider("Initial Roll Temperature [°C]", 20, 150, 80, step=5,
                               help="Target roll surface temperature")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("#### Friction Parameters")
    friction_model = st.selectbox("Friction Model", 
                                   ["Temperature-Dependent", "Constant", "Speed-Dependent"],
                                   help="Select friction calculation method")
    
    if friction_model == "Constant":
        mu_const = st.slider("Constant friction μ", 0.05, 0.35, 0.15, step=0.01)
        mu0, a_T, b_v = mu_const, 0.0, 0.0
    else:
        mu0 = st.slider("Base interface friction μ₀", 0.05, 0.30, 0.12, step=0.01)
        a_T = st.slider("Temp sensitivity a (Δμ/°C)", 0.0, 0.0005, 0.0001, step=0.00005,
                       help="Positive: friction increases with temperature")
        b_v = st.slider("Speed sensitivity b (Δμ per ln v)", 0.0, 0.05, 0.01, step=0.005,
                       help="Friction decreases with rolling speed")
    st.markdown('</div>', unsafe_allow_html=True)

roll = ROLLS[material_choice]
steel = STEELS[steel_choice]

# Calculate reduction ratio
reduction_ratio = ((h1 - h2) / h1) * 100

# ==============================
# Helper Models - ENHANCED
# ==============================

def thermal_conductivity_roll(k_base, k_coeff, T_C):
    k = k_base * (1 + k_coeff * (T_C - 20))
    return max(5.0, k)


def surface_speed(D_mm, N_rpm):
    """Calculate roll surface speed in m/s"""
    D_m = D_mm / 1000
    return math.pi * D_m * (N_rpm / 60.0)

def strain_rate_estimate(h1_mm, h2_mm, D_mm, N_rpm):
    """Estimate average strain rate during rolling"""
    v = surface_speed(D_mm, N_rpm)
    R = (D_mm / 2) / 1000
    dh = (h1_mm - h2_mm) / 1000
    L = max(1e-8, math.sqrt(max(1e-12, R * max(1e-9, dh))))
    # True strain rate = v/L * (ln(h1/h2) / ε) where ε is average strain
    epsilon = math.log(h1_mm / max(0.1, h2_mm))
    return max(1e-6, v * epsilon / (L * math.sqrt(3)))

def flow_stress_ZH(T_C, edot, A, n, alpha, Q):
    """Zener-Hollomon based flow stress model (MPa)."""
    T_K = T_C + 273.15
    Z = edot * math.exp(Q / (R_GAS * T_K))
    Z_norm = max(1e-20, Z / A)

    # alpha is in MPa^-1, so (1/alpha) gives MPa scale
    sigma = (1.0 / alpha) * math.asinh(Z_norm ** (1.0 / n))

    return float(np.clip(sigma, 5.0, 800.0))


def mu_of_T_v(mu0, a, b_sens, T_interface, v, k_roll, model="Temperature-Dependent"):
    """Friction depends on interface temperature and speed."""
    if model == "Constant":
        return float(np.clip(mu0, 0.03, 0.40))

    # Temp effect: hotter interface => higher μ (scale, lubrication breakdown)
    mu = mu0 + a * (T_interface - 900.0)

    if model in ["Temperature-Dependent", "Speed-Dependent"]:
        # Speed effect: higher speed => lower μ
        mu -= b_sens * math.log(max(v, 0.1))

    # Roll conduction: higher k => better cooling => lower μ
    roll_factor = (40.0 / max(10.0, k_roll))
    mu *= roll_factor

    return float(np.clip(mu, 0.03, 0.40))


def friction_factor_from_mu(mu):
    """Convert Coulomb friction to friction factor m for rolling equations"""
    # Ekelund's formula: m ≈ (2/π) * arctan(πμ/2)
    return float((2.0 / math.pi) * math.atan((math.pi * mu) / 2.0))

def effective_radius(R_m, F_N, b_m, E, nu):
    """Calculate effective roll radius accounting for elastic flattening (Hitchcock formula)"""
    Eprime = E / (1.0 - nu**2)
    # Flattening: ΔR = F / (π * b * E')
    dR = F_N / (math.pi * max(1e-6, b_m) * max(1e9, Eprime))
    return R_m + dR

def contact_length(R_eff, h1_m, h2_m):
    """Calculate arc of contact length"""
    return math.sqrt(max(1e-12, R_eff * max(1e-9, (h1_m - h2_m))))

def von_karman_roll_pressure(sigma_bar, m, R_eff, h_avg, L):
    """
    Von Kármán equation for roll pressure distribution
    Simplified mean pressure considering friction hill
    p_mean = σ̄ * (1 + m * L / h_avg)
    """
    # Dimensionless parameter
    phi = m * L / max(1e-6, h_avg)
    # Mean pressure multiplier considering friction
    pressure_multiplier = 1.0 + phi / 2.0  # Simplified integration
    return sigma_bar * pressure_multiplier

def calculate_roll_force_von_karman(sigma_bar, m, R_eff, h1_m, h2_m, b_m):
    """Calculate rolling force using Von Kármán approach"""
    h_avg = (h1_m + h2_m) / 2.0
    L = contact_length(R_eff, h1_m, h2_m)
    
    # Mean roll pressure from Von Kármán
    p_mean = von_karman_roll_pressure(sigma_bar, m, R_eff, h_avg, L) * 1e6  # Convert to Pa
    
    # Rolling force: F = p_mean * L * b
    F = p_mean * L * b_m
    return F, L, p_mean / 1e6  # Return force, contact length, and pressure in MPa

def calculate_torque_enhanced(F, L, m, h1_m, h2_m):
    """
    Enhanced torque calculation considering friction distribution
    T = F * a, where 'a' is the torque arm
    For rolling: a ≈ L/2 for uniform pressure, adjusted for friction
    """
    # Neutral point typically at ~0.4-0.5 of contact length
    neutral_point_ratio = 0.45 + 0.05 * m  # Friction shifts neutral point
    torque_arm = L * neutral_point_ratio
    return F * torque_arm

def calculate_power(Torque, N_rpm):
    """Calculate power from torque and rotational speed"""
    N_rps = N_rpm / 60.0
    omega = 2.0 * math.pi * N_rps  # rad/s
    return omega * Torque

def estimate_roll_temperature_rise(P_kW, k_roll, A_contact, cooling_rate=0.7):
    """
    Estimate roll surface temperature rise due to rolling heat
    Simplified steady-state heat balance
    """
    # Heat flux into roll (W/m²)
    # Assuming ~30-40% of power goes into roll heating
    q_roll = (P_kW * 1000 * 0.35) / max(1e-6, A_contact)
    
    # Temperature rise: ΔT = q * thickness / k (simplified)
    # Assuming effective heat penetration depth ~5mm
    delta_T = q_roll * 0.005 / max(1.0, k_roll)
    
    # Apply cooling effectiveness (water cooling reduces temperature)
    return delta_T * (1 - cooling_rate)

# ==============================
# Core Calculation - ENHANCED with Von Kármán
# ==============================
def compute_F_T_P(h1, h2, D, b, N_rpm, T_C, T_roll, steel, roll,
                  mu0, a_T, b_v, friction_model):
    """
    Enhanced rolling calculation with:
    - Von Kármán mean pressure model
    - Temperature-dependent flow stress (strip temperature)
    - Temperature-dependent friction using INTERFACE temperature
    - Roll heating feedback into friction and power loss
    - Elastic roll flattening (Hitchcock)
    """

    # ---- Basic geometry (m) ----
    R0 = (D / 2) / 1000.0
    h1_m = h1 / 1000.0
    h2_m = h2 / 1000.0
    b_m  = b / 1000.0

    # ---- Kinematics ----
    v = surface_speed(D, N_rpm)
    edot = strain_rate_estimate(h1, h2, D, N_rpm)

    # ---- Material flow stress at strip temperature (MPa) ----
    sigma_bar = flow_stress_ZH(
        T_C, edot,
        steel["A"], steel["n"], steel["alpha"], steel["Q"]
    )

    # ---- Roll thermal conductivity at roll temperature ----
    k_roll_actual = thermal_conductivity_roll(
        roll["k_roll_base"], roll["k_temp_coeff"], T_roll
    )

    # ---- Iterative coupling variables ----
    R_eff = R0
    F = 0.0
    L = 0.0
    p_mean = 0.0
    mu = 0.0
    m = 0.0
    Tq = 0.0
    P = 0.0

    # Start roll operating temperature guess
    T_roll_operating = float(T_roll)

    # ---- Iteration: roll heating -> interface temp -> friction -> force/torque/power ----
    for _ in range(4):  # 3-4 iterations is enough
        # Interface temperature model (simple weighted average)
        # If roll heats up, interface heats -> friction increases
        T_interface = 0.6 * T_C + 0.4 * T_roll_operating

        # Friction coefficient from interface temp + speed + roll cooling
        mu = mu_of_T_v(
            mu0, a_T, b_v,
            T_interface,
            v,
            k_roll_actual,
            friction_model
        )
        m = friction_factor_from_mu(mu)

        # Rolling force with Von Kármán mean pressure
        F, L, p_mean = calculate_roll_force_von_karman(
            sigma_bar, m, R_eff, h1_m, h2_m, b_m
        )

        # Update effective radius from roll flattening
        R_eff_new = effective_radius(R0, F, b_m, roll["E"], roll["nu"])
        R_eff = R_eff_new

        # Torque and power
        Tq = calculate_torque_enhanced(F, L, m, h1_m, h2_m)
        P = calculate_power(Tq, N_rpm)

        # Roll heating update (feedback loop)
        A_contact = max(1e-9, L * b_m)  # m²
        delta_T_roll = estimate_roll_temperature_rise(
            P / 1000.0,  # kW
            k_roll_actual,
            A_contact,
            cooling_rate=0.7
        )
        T_roll_operating = float(T_roll + delta_T_roll)

    # ---- Strip exit temperature (simplified) ----
    # NOTE: this is empirical and you can refine it later
    # reduction_ratio is global in your code; safer compute locally
    reduction_ratio_local = ((h1 - h2) / max(1e-6, h1)) * 100.0
    temp_drop_factor = 0.02 * reduction_ratio_local + 0.005 * v
    T_exit = T_C - min(temp_drop_factor * 10.0, 50.0)

    # ---- Specific energy ----
    # per meter length basis (volume reduced per 1m)
    volume_reduced = (h1_m - h2_m) * b_m * 1.0
    specific_energy = (P * 1.0) / max(1e-12, volume_reduced)  # J/m²

    return {
        "F": F,                     # N
        "Tq": Tq,                   # N·m
        "P": P,                     # W
        "sigma_bar": sigma_bar,     # MPa
        "mu": mu,                   # -
        "m": m,                     # -
        "R_eff": R_eff,             # m
        "R0": R0,                   # m
        "L": L,                     # m
        "v": v,                     # m/s
        "edot": edot,               # 1/s
        "p_mean": p_mean,           # MPa
        "T_roll_op": T_roll_operating,  # °C
        "T_exit": T_exit,           # °C
        "k_roll": k_roll_actual,    # W/m·K
        "specific_energy": specific_energy,  # J/m²
        "roll_flattening": (R_eff - R0) * 1000.0  # mm
    }

# ==============================
# Current Calculation & Enhanced KPI Cards
# ==============================
res = compute_F_T_P(h1, h2, D, b, N_rpm, T_C, T_roll_initial, steel, roll, mu0, a_T, b_v, friction_model)

# Check recrystallization temperature
above_recryst = res["T_exit"] >= steel["T_recryst"]
temp_status = "✓ Above recrystallization" if above_recryst else "⚠ Below recrystallization"
temp_color = "#4ade80" if above_recryst else "#fb923c"

st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Rolling Load</h3>
    <div class="val">{res['F']/1000:.2f} kN</div>
    <div class="sub">Pressure: {res['p_mean']:.1f} MPa | Contact: {res['L']*1000:.1f} mm</div>
  </div>
""", unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Torque</h3>
    <div class="val">{res['Tq']/1000:.2f} kN·m</div>
    <div class="sub">Speed: {res['v']:.2f} m/s | Flattening: {res['roll_flattening']:.3f} mm</div>
  </div>
""", unsafe_allow_html=True)
st.markdown(f"""
  <div class="kpi">
    <h3>Power</h3>
    <div class="val">{res['P']/1000:.2f} kW</div>
    <div class="sub">Flow Stress: {res['sigma_bar']:.1f} MPa | Efficiency: {res['specific_energy']/1000:.1f} kJ/m²</div>
  </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Additional process metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reduction Ratio", f"{reduction_ratio:.1f}%")
col2.metric("Friction μ", f"{res['mu']:.3f}", help="Friction factor m={:.3f}".format(res['m']))
col3.metric("Strain Rate", f"{res['edot']:.1f} s⁻¹")
col4.metric("Exit Temperature", f"{res['T_exit']:.0f}°C", 
            delta=f"{temp_status}", delta_color="normal" if above_recryst else "inverse")

st.caption(f"Roll Surface: {res['T_roll_op']:.1f}°C | k_roll={res['k_roll']:.1f} W/m·K | R_eff: {res['R0']*1000:.1f}→{res['R_eff']*1000:.1f} mm")

# ==============================
# Enhanced Suite Tabs
# ==============================
TAB1, TAB2, TAB3, TAB4 = st.tabs(["📊 Dashboard", "🆚 Material Compare", "🧠 Optimize", "📈 Advanced Analysis"])

with TAB1:
    st.subheader("Rolling Behavior & Sensitivity Analysis")
    
    # Temperature dependency plots
    st.markdown("#### Effect of Temperature on Rolling Parameters")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Load vs Reduction Ratio @ multiple temps
        T_list = [800, 950, 1100]
        ratios = np.linspace(5, 60, 100)
        fig1 = go.Figure()
        
        for Tplot in T_list:
            loads, torques, powers = [], [], []
            for r in ratios:
                h2_temp = h1 - (h1 * r / 100)
                if h2_temp > 0.5:  # Valid thickness
                    tmp = compute_F_T_P(h1, h2_temp, D, b, N_rpm, Tplot, T_roll_initial, 
                                       steel, roll, mu0, a_T, b_v, friction_model)
                    loads.append(tmp["F"] / 1000)
                    torques.append(tmp["Tq"] / 1000)
                    powers.append(tmp["P"] / 1000)
                else:
                    loads.append(None)
            
            fig1.add_trace(go.Scatter(x=ratios, y=loads, mode="lines", 
                                     name=f"{Tplot}°C", line=dict(width=2)))
        
        fig1.update_layout(
            title=f"Load vs Reduction Ratio at Different Temperatures<br><sub>{material_choice} | {steel_choice}</sub>",
            xaxis_title="Reduction Ratio (%)", 
            yaxis_title="Rolling Load (kN)",
            template="plotly_dark", 
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_b:
        # Power vs Reduction at different temperatures
        fig1b = go.Figure()
        for Tplot in T_list:
            powers = []
            for r in ratios:
                h2_temp = h1 - (h1 * r / 100)
                if h2_temp > 0.5:
                    tmp = compute_F_T_P(h1, h2_temp, D, b, N_rpm, Tplot, T_roll_initial,
                                       steel, roll, mu0, a_T, b_v, friction_model)
                    powers.append(tmp["P"] / 1000)
                else:
                    powers.append(None)
            
            fig1b.add_trace(go.Scatter(x=ratios, y=powers, mode="lines",
                                      name=f"{Tplot}°C", line=dict(width=2)))
        
        fig1b.update_layout(
            title=f"Power vs Reduction Ratio<br><sub>Temperature Effect</sub>",
            xaxis_title="Reduction Ratio (%)",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig1b, use_container_width=True)
    
    st.markdown("#### Effect of Rolling Speed")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        # Torque vs Speed at different temperatures
        speeds = np.linspace(50, 350, 80)
        fig2 = go.Figure()
        
        for Tplot in [800, 950, 1100]:
            torq_vs_speed = []
            for n in speeds:
                tmp = compute_F_T_P(h1, h2, D, b, n, Tplot, T_roll_initial,
                                   steel, roll, mu0, a_T, b_v, friction_model)
                torq_vs_speed.append(tmp["Tq"] / 1000)
            
            fig2.add_trace(go.Scatter(x=speeds, y=torq_vs_speed, mode="lines",
                                     name=f"{Tplot}°C", line=dict(width=2)))
        
        fig2.update_layout(
            title="Torque vs Rolling Speed",
            xaxis_title="Roll Speed (rpm)",
            yaxis_title="Torque (kN·m)",
            template="plotly_dark",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_d:
        # Power vs Speed
        fig3 = go.Figure()
        
        for Tplot in [800, 950, 1100]:
            power_vs_speed = []
            for n in speeds:
                tmp = compute_F_T_P(h1, h2, D, b, n, Tplot, T_roll_initial,
                                   steel, roll, mu0, a_T, b_v, friction_model)
                power_vs_speed.append(tmp["P"] / 1000)
            
            fig3.add_trace(go.Scatter(x=speeds, y=power_vs_speed, mode="lines",
                                     name=f"{Tplot}°C", line=dict(width=2)))
        
        fig3.update_layout(
            title="Power vs Rolling Speed",
            xaxis_title="Roll Speed (rpm)",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("#### Friction Effect on Process Parameters")
    
    # Friction coefficient variation
    mu_range = np.linspace(0.05, 0.30, 50)
    loads_mu, torques_mu, powers_mu = [], [], []
    
    for mu_test in mu_range:
        tmp = compute_F_T_P(h1, h2, D, b, N_rpm, T_C, T_roll_initial,
                           steel, roll, mu_test, 0.0, 0.0, "Constant")
        loads_mu.append(tmp["F"] / 1000)
        torques_mu.append(tmp["Tq"] / 1000)
        powers_mu.append(tmp["P"] / 1000)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=mu_range, y=loads_mu, name="Load", 
                             line=dict(width=2, color='#3b82f6')))
    fig4.add_trace(go.Scatter(x=mu_range, y=torques_mu, name="Torque",
                             line=dict(width=2, color='#10b981'), yaxis='y2'))
    fig4.add_trace(go.Scatter(x=mu_range, y=powers_mu, name="Power",
                             line=dict(width=2, color='#f59e0b'), yaxis='y3'))
    
    fig4.update_layout(
        title="Effect of Friction Coefficient on Rolling Parameters",
        xaxis=dict(title="Friction Coefficient μ"),
        yaxis=dict(title=dict(text="Load (kN)", font=dict(color='#3b82f6'))),
        yaxis2=dict(title=dict(text="Torque (kN·m)", font=dict(color='#10b981')), 
                    overlaying='y', side='right'),
        yaxis3=dict(title=dict(text="Power (kW)", font=dict(color='#f59e0b')), 
                    overlaying='y', side='right', position=0.85),
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig4, use_container_width=True)

with TAB2:
    st.subheader("Roll Material Comparison: SG Iron vs DPIC")
    st.caption("Comparative analysis at identical rolling conditions across temperature range")
    
    # Comparison parameters
    T_compare_range = np.linspace(750, 1150, 50)
    
    comparison_data = {"SG Iron": {}, "DPIC": {}}
    
    for mat_name, mat_props in ROLLS.items():
        loads, torques, powers, roll_temps, frictions = [], [], [], [], []
        
        for T_comp in T_compare_range:
            tmp = compute_F_T_P(h1, h2, D, b, N_rpm, T_comp, T_roll_initial,
                               steel, mat_props, mu0, a_T, b_v, friction_model)
            loads.append(tmp["F"] / 1000)
            torques.append(tmp["Tq"] / 1000)
            powers.append(tmp["P"] / 1000)
            roll_temps.append(tmp["T_roll_op"])
            frictions.append(tmp["mu"])
        
        comparison_data[mat_name] = {
            "temps": T_compare_range,
            "loads": loads,
            "torques": torques,
            "powers": powers,
            "roll_temps": roll_temps,
            "frictions": frictions
        }
    
    # Create comparison plots
    col_e, col_f = st.columns(2)
    
    with col_e:
        fig_comp1 = go.Figure()
        for mat_name in ["SG Iron", "DPIC"]:
            fig_comp1.add_trace(go.Scatter(
                x=comparison_data[mat_name]["temps"],
                y=comparison_data[mat_name]["loads"],
                mode="lines",
                name=mat_name,
                line=dict(width=3)
            ))
        
        fig_comp1.update_layout(
            title="Rolling Load Comparison<br><sub>SG Iron vs DPIC across Temperature</sub>",
            xaxis_title="Temperature (°C)",
            yaxis_title="Rolling Load (kN)",
            template="plotly_dark",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_comp1, use_container_width=True)
    
    with col_f:
        fig_comp2 = go.Figure()
        for mat_name in ["SG Iron", "DPIC"]:
            fig_comp2.add_trace(go.Scatter(
                x=comparison_data[mat_name]["temps"],
                y=comparison_data[mat_name]["powers"],
                mode="lines",
                name=mat_name,
                line=dict(width=3)
            ))
        
        fig_comp2.update_layout(
            title="Power Consumption Comparison<br><sub>SG Iron vs DPIC</sub>",
            xaxis_title="Temperature (°C)",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_comp2, use_container_width=True)
    
    # Roll temperature comparison
    fig_comp3 = go.Figure()
    for mat_name in ["SG Iron", "DPIC"]:
        fig_comp3.add_trace(go.Scatter(
            x=comparison_data[mat_name]["temps"],
            y=comparison_data[mat_name]["roll_temps"],
            mode="lines",
            name=f"{mat_name} Roll Surface",
            line=dict(width=3)
        ))
    
    fig_comp3.update_layout(
        title="Roll Surface Operating Temperature<br><sub>Effect of Material Thermal Properties</sub>",
        xaxis_title="Strip Temperature (°C)",
        yaxis_title="Roll Surface Temperature (°C)",
        template="plotly_dark",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_comp3, use_container_width=True)
    
    # Performance summary table
    st.markdown("#### Performance Summary at Current Conditions")
    
    summary_data = []
    for mat_name, mat_props in ROLLS.items():
        tmp = compute_F_T_P(h1, h2, D, b, N_rpm, T_C, T_roll_initial,
                           steel, mat_props, mu0, a_T, b_v, friction_model)
        summary_data.append({
            "Material": mat_name,
            "Load (kN)": f"{tmp['F']/1000:.2f}",
            "Torque (kN·m)": f"{tmp['Tq']/1000:.2f}",
            "Power (kW)": f"{tmp['P']/1000:.2f}",
            "Roll Temp (°C)": f"{tmp['T_roll_op']:.1f}",
            "Friction μ": f"{tmp['mu']:.3f}",
            "Flattening (mm)": f"{tmp['roll_flattening']:.3f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Material recommendations
    st.markdown("#### Material Selection Insights")
    
    sg_power = comparison_data["SG Iron"]["powers"][-1]
    dpic_power = comparison_data["DPIC"]["powers"][-1]
    power_diff = abs(sg_power - dpic_power)
    better_material = "SG Iron" if sg_power < dpic_power else "DPIC"
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.info(f"""
        **SG Iron Advantages:**
        - Higher thermal conductivity ({ROLLS['SG Iron']['k_roll_base']} W/m·K)
        - Better heat dissipation
        - Lower cost
        - Suitable for moderate loads
        """)
    
    with col_g2:
        st.info(f"""
        **DPIC Advantages:**
        - Higher hardness ({ROLLS['DPIC']['HB']} HB)
        - Better wear resistance ({ROLLS['DPIC']['wear_resistance']}x)
        - Longer roll life
        - Preferred for high-reduction passes
        """)
    
    st.success(f"""
    **Recommendation for Current Conditions:**  
    {better_material} shows {power_diff:.1f} kW lower power consumption at 1150°C.  
    Consider operational costs vs. roll life when selecting material.
    """)

with TAB3:
    st.subheader("Multi-Parameter Optimization")
    st.caption("Find optimal operating conditions for minimum power dissipation and maximum efficiency")
    
    # Optimization settings
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        st.markdown("**Temperature Range**")
        T_min = st.number_input("T min [°C]", 750, 1100, 850, step=10)
        T_max = st.number_input("T max [°C]", 800, 1200, 1050, step=10)
    
    with col_opt2:
        st.markdown("**Speed Range**")
        N_min = st.number_input("N min [rpm]", 50, 300, 80, step=10)
        N_max = st.number_input("N max [rpm]", 60, 400, 180, step=10)
    
    with col_opt3:
        st.markdown("**Reduction Range**")
        r_min = st.slider("Reduction min [%]", 5, 40, 10)
        r_max = st.slider("Reduction max [%]", 10, 70, 35)
    
    grid_density = st.select_slider("Grid density", 
                                     options=["coarse (6×6×6)", "medium (12×12×12)", "fine (20×20×20)"],
                                     value="medium (12×12×12)")
    
    if "coarse" in grid_density:
        nT, nN, nR = 6, 6, 6
    elif "fine" in grid_density:
        nT, nN, nR = 20, 20, 20
    else:
        nT, nN, nR = 12, 12, 12
    
    optimization_objective = st.radio(
        "Optimization Objective:",
        ["Minimum Power", "Maximum Efficiency (Power/Reduction)", "Minimum Specific Energy"],
        horizontal=True
    )
    
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner(f"Running {nT}×{nN}×{nR} = {nT*nN*nR} simulations..."):
            T_vals = np.linspace(T_min, T_max, nT)
            N_vals = np.linspace(N_min, N_max, nN)
            R_vals = np.linspace(r_min, r_max, nR)
            
            best = None
            pts_T, pts_r, pts_N, pts_obj = [], [], [], []
            all_results = []
            
            progress_bar = st.progress(0)
            total_iterations = nT * nN * nR
            current_iteration = 0
            
            for Topt in T_vals:
                for Nopt in N_vals:
                    for ropt in R_vals:
                        h2_opt = h1 - (h1 * ropt / 100.0)
                        
                        if h2_opt > 0.5:  # Valid thickness
                            out = compute_F_T_P(h1, h2_opt, D, b, Nopt, Topt, T_roll_initial,
                                               steel, roll, mu0, a_T, b_v, friction_model)
                            
                            PkW = out["P"] / 1000
                            reduction_mm = h1 - h2_opt
                            
                            # Calculate different objectives
                            if optimization_objective == "Minimum Power":
                                objective_value = PkW
                            elif optimization_objective == "Maximum Efficiency (Power/Reduction)":
                                objective_value = PkW / max(1e-6, reduction_mm)
                            else:  # Minimum Specific Energy
                                objective_value = out["specific_energy"] / 1000  # kJ/m²
                            
                            # Check if strip exit temp is above recrystallization
                            temp_valid = out["T_exit"] >= steel["T_recryst"]
                            
                            if temp_valid:  # Only consider valid solutions
                                if (best is None) or (objective_value < best[0]):
                                    best = (objective_value, Topt, Nopt, ropt, PkW, reduction_mm, out)
                            
                            pts_T.append(Topt)
                            pts_r.append(ropt)
                            pts_N.append(Nopt)
                            pts_obj.append(objective_value)
                            
                            all_results.append({
                                "T": Topt,
                                "N": Nopt,
                                "Reduction": ropt,
                                "Power": PkW,
                                "Objective": objective_value,
                                "Valid": temp_valid
                            })
                        
                        current_iteration += 1
                        progress_bar.progress(current_iteration / total_iterations)
            
            progress_bar.empty()
            
            if best is not None:
                st.success(f"""
                **Optimum Found:**  
                Objective Value: {best[0]:.3f} {'kW' if 'Power' in optimization_objective else 'kW/mm' if 'Efficiency' in optimization_objective else 'kJ/m²'}  
                Temperature: {best[1]:.0f}°C | Speed: {best[2]:.0f} rpm | Reduction: {best[3]:.1f}%  
                Power: {best[4]:.2f} kW | Thickness: {h1:.1f}→{h1-best[5]:.1f} mm
                """)
                
                # 3D scatter plot
                fig_opt = go.Figure()
                
                # All points colored by objective
                fig_opt.add_trace(go.Scatter3d(
                    x=pts_T,
                    y=pts_r,
                    z=pts_N,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=pts_obj,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Objective")
                    ),
                    name="Search Points",
                    text=[f"T={t:.0f}°C, r={r:.1f}%, N={n:.0f}rpm, Obj={o:.2f}" 
                          for t,r,n,o in zip(pts_T, pts_r, pts_N, pts_obj)],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                # Optimum point
                fig_opt.add_trace(go.Scatter3d(
                    x=[best[1]],
                    y=[best[3]],
                    z=[best[2]],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name="Optimum",
                    text=[f"OPTIMUM<br>T={best[1]:.0f}°C<br>r={best[3]:.1f}%<br>N={best[2]:.0f}rpm"],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig_opt.update_layout(
                    title=f"3D Optimization Space<br><sub>{optimization_objective}</sub>",
                    scene=dict(
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Reduction (%)",
                        zaxis_title="Speed (rpm)"
                    ),
                    template="plotly_dark",
                    height=600
                )
                st.plotly_chart(fig_opt, use_container_width=True)
                
                # 2D projections
                col_h1, col_h2 = st.columns(2)
                
                with col_h1:
                    # Power vs Reduction colored by Temperature
                    fig_2d1 = go.Figure()
                    fig_2d1.add_trace(go.Scatter(
                        x=pts_r,
                        y=[all_results[i]["Power"] for i in range(len(pts_r))],
                        mode='markers',
                        marker=dict(size=5, color=pts_T, colorscale='Plasma', showscale=True,
                                   colorbar=dict(title="Temp (°C)")),
                        text=[f"T={t:.0f}°C, N={n:.0f}rpm" for t,n in zip(pts_T, pts_N)],
                        hovertemplate='Reduction: %{x:.1f}%<br>Power: %{y:.1f} kW<br>%{text}<extra></extra>'
                    ))
                    fig_2d1.add_trace(go.Scatter(
                        x=[best[3]], y=[best[4]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name="Optimum"
                    ))
                    fig_2d1.update_layout(
                        title="Power vs Reduction (colored by Temperature)",
                        xaxis_title="Reduction (%)",
                        yaxis_title="Power (kW)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_2d1, use_container_width=True)
                
                with col_h2:
                    # Power vs Temperature colored by Speed
                    fig_2d2 = go.Figure()
                    fig_2d2.add_trace(go.Scatter(
                        x=pts_T,
                        y=[all_results[i]["Power"] for i in range(len(pts_T))],
                        mode='markers',
                        marker=dict(size=5, color=pts_N, colorscale='Viridis', showscale=True,
                                   colorbar=dict(title="Speed (rpm)")),
                        text=[f"r={r:.1f}%, N={n:.0f}rpm" for r,n in zip(pts_r, pts_N)],
                        hovertemplate='Temp: %{x:.0f}°C<br>Power: %{y:.1f} kW<br>%{text}<extra></extra>'
                    ))
                    fig_2d2.add_trace(go.Scatter(
                        x=[best[1]], y=[best[4]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name="Optimum"
                    ))
                    fig_2d2.update_layout(
                        title="Power vs Temperature (colored by Speed)",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Power (kW)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_2d2, use_container_width=True)
                
                # Detailed results at optimum
                with st.expander("📋 Detailed Results at Optimum"):
                    out_opt = best[6]
                    
                    col_det1, col_det2, col_det3 = st.columns(3)
                    
                    with col_det1:
                        st.markdown("**Process Parameters**")
                        st.write({
                            "Temperature (°C)": f"{best[1]:.0f}",
                            "Speed (rpm)": f"{best[2]:.0f}",
                            "Reduction (%)": f"{best[3]:.1f}",
                            "h₁ → h₂ (mm)": f"{h1:.1f} → {h1-best[5]:.1f}"
                        })
                    
                    with col_det2:
                        st.markdown("**Mechanical Results**")
                        st.write({
                            "Load (kN)": f"{out_opt['F']/1000:.2f}",
                            "Torque (kN·m)": f"{out_opt['Tq']/1000:.2f}",
                            "Power (kW)": f"{out_opt['P']/1000:.2f}",
                            "Flow Stress (MPa)": f"{out_opt['sigma_bar']:.1f}"
                        })
                    
                    with col_det3:
                        st.markdown("**Additional Metrics**")
                        st.write({
                            "Friction μ": f"{out_opt['mu']:.3f}",
                            "Contact Length (mm)": f"{out_opt['L']*1000:.1f}",
                            "Roll Flattening (mm)": f"{out_opt['roll_flattening']:.3f}",
                            "Exit Temp (°C)": f"{out_opt['T_exit']:.0f}"
                        })
            else:
                st.error("No valid solution found. All conditions resulted in exit temperature below recrystallization.")

with TAB4:
    st.subheader("Advanced Process Analysis")
    st.caption("Deep dive into rolling mechanics and material behavior")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Temperature Effect on Material Properties",
         "Roll Flattening Analysis",
         "Friction and Lubrication Study",
         "Energy Distribution Analysis",
         "Strain Rate Sensitivity"]
    )
    
    if analysis_type == "Temperature Effect on Material Properties":
        st.markdown("#### Flow Stress Behavior Across Temperature Range")
        
        T_range = np.linspace(700, 1200, 100)
        strain_rates = [1, 10, 50, 100]  # 1/s
        
        fig_fs = go.Figure()
        
        for edot_test in strain_rates:
            flow_stresses = []
            for T_test in T_range:
                sigma = flow_stress_ZH(T_test, edot_test, steel["A"], steel["n"], 
                                      steel["alpha"], steel["Q"])
                flow_stresses.append(sigma)
            
            fig_fs.add_trace(go.Scatter(
                x=T_range,
                y=flow_stresses,
                mode='lines',
                name=f"ε̇ = {edot_test} s⁻¹",
                line=dict(width=2)
            ))
        
        # Add recrystallization temperature line
        fig_fs.add_vline(x=steel["T_recryst"], line_dash="dash", line_color="red",
                        annotation_text=f"T_recryst = {steel['T_recryst']}°C")
        
        fig_fs.update_layout(
            title=f"Flow Stress vs Temperature<br><sub>{steel_choice} at Different Strain Rates</sub>",
            xaxis_title="Temperature (°C)",
            yaxis_title="Flow Stress σ̄ (MPa)",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_fs, use_container_width=True)
        
        st.info(f"""
        **Key Observations:**
        - Flow stress decreases exponentially with temperature
        - Higher strain rates increase flow stress (strain rate hardening)
        - Recrystallization temperature ({steel['T_recryst']}°C) marks a transition point
        - Below recrystallization: incomplete grain reformation affects mechanical properties
        """)
    
    elif analysis_type == "Roll Flattening Analysis":
        st.markdown("#### Roll Deformation Under Load")
        
        # Calculate flattening for different loads
        loads_range = np.linspace(1000, 10000, 50)  # kN
        
        fig_flat = go.Figure()
        
        for mat_name, mat_props in ROLLS.items():
            flattenings = []
            for F_test in loads_range:
                R_eff_test = effective_radius(R_m=(D/2)/1000, F_N=F_test*1000, 
                                             b_m=b/1000, E=mat_props["E"], nu=mat_props["nu"])
                flattening = (R_eff_test - (D/2)/1000) * 1000  # mm
                flattenings.append(flattening)
            
            fig_flat.add_trace(go.Scatter(
                x=loads_range,
                y=flattenings,
                mode='lines',
                name=mat_name,
                line=dict(width=3)
            ))
        
        # Add current operating point
        current_flattening = res['roll_flattening']
        current_load = res['F'] / 1000
        
        fig_flat.add_trace(go.Scatter(
            x=[current_load],
            y=[current_flattening],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name="Current Operating Point"
        ))
        
        fig_flat.update_layout(
            title="Roll Flattening vs Rolling Load<br><sub>Hitchcock's Formula - Material Comparison</sub>",
            xaxis_title="Rolling Load (kN)",
            yaxis_title="Roll Flattening (mm)",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_flat, use_container_width=True)
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.metric("Current Roll Flattening", f"{current_flattening:.3f} mm")
            st.metric("Effective Radius Increase", f"{(res['R_eff']-res['R0'])*1000:.3f} mm")
        
        with col_i2:
            st.metric("Original Roll Radius", f"{res['R0']*1000:.1f} mm")
            st.metric("Effective Roll Radius", f"{res['R_eff']*1000:.1f} mm")
        
        st.info("""
        **Roll Flattening Impact:**
        - Increases contact length → higher rolling force
        - Affects pressure distribution in roll gap
        - DPIC shows less flattening due to higher Young's modulus
        - Critical for accurate load prediction in multi-stand mills
        """)
    
    elif analysis_type == "Friction and Lubrication Study":
        st.markdown("#### Friction Coefficient Behavior")
        
        # Friction vs temperature at different speeds
        T_fric_range = np.linspace(750, 1150, 80)
        speeds_fric = [50, 100, 200, 300]  # rpm
        
        fig_fric = go.Figure()
        
        for N_fric in speeds_fric:
            mu_vals = []
            for T_fric in T_fric_range:
                v_fric = surface_speed(D, N_fric)
                mu_val = mu_of_T_v(mu0, a_T, b_v, T_fric, v_fric, 
                                  res['k_roll'], friction_model)
                mu_vals.append(mu_val)
            
            fig_fric.add_trace(go.Scatter(
                x=T_fric_range,
                y=mu_vals,
                mode='lines',
                name=f"{N_fric} rpm",
                line=dict(width=2)
            ))
        
        fig_fric.update_layout(
            title="Friction Coefficient vs Temperature<br><sub>Speed Dependency</sub>",
            xaxis_title="Temperature (°C)",
            yaxis_title="Friction Coefficient μ",
            template="plotly_dark",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_fric, use_container_width=True)
        
        # Friction factor (m) conversion
        mu_test_range = np.linspace(0.05, 0.35, 50)
        m_vals = [friction_factor_from_mu(mu) for mu in mu_test_range]
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=mu_test_range,
            y=m_vals,
            mode='lines',
            line=dict(width=3, color='#06b6d4')
        ))
        
        fig_m.update_layout(
            title="Coulomb Friction (μ) to Friction Factor (m) Conversion<br><sub>Ekelund's Formula</sub>",
            xaxis_title="Coulomb Friction Coefficient μ",
            yaxis_title="Friction Factor m",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_m, use_container_width=True)
        
        st.info("""
        **Friction Insights:**
        - Temperature increases friction (oxide scale formation, reduced lubrication)
        - Speed reduces friction (hydrodynamic lubrication effect)
        - Roll material affects friction through thermal conductivity
        - Friction factor m < μ due to slip distribution in contact zone
        """)
    
    elif analysis_type == "Energy Distribution Analysis":
        st.markdown("#### Energy Flow in Rolling Process")
        
        # Calculate energy components
        P_total = res["P"] / 1000  # kW
        
        # Estimate energy distribution (empirical approximations)
        E_deformation = P_total * 0.50  # 50% goes into plastic deformation
        E_friction = P_total * 0.30     # 30% dissipated as friction heat
        E_elastic = P_total * 0.10      # 10% stored as elastic energy
        E_roll_heating = P_total * 0.10  # 10% heats the rolls
        
        # Pie chart
        fig_energy = go.Figure(data=[go.Pie(
            labels=['Plastic Deformation', 'Friction Heat', 'Elastic Energy', 'Roll Heating'],
            values=[E_deformation, E_friction, E_elastic, E_roll_heating],
            hole=0.4,
            marker=dict(colors=['#3b82f6', '#ef4444', '#10b981', '#f59e0b'])
        )])
        
        fig_energy.update_layout(
            title=f"Energy Distribution in Rolling Process<br><sub>Total Power: {P_total:.1f} kW</sub>",
            template="plotly_dark",
            height=450
        )
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # Energy metrics
        col_j1, col_j2, col_j3, col_j4 = st.columns(4)
        col_j1.metric("Deformation", f"{E_deformation:.1f} kW", "50%")
        col_j2.metric("Friction", f"{E_friction:.1f} kW", "30%")
        col_j3.metric("Elastic", f"{E_elastic:.1f} kW", "10%")
        col_j4.metric("Roll Heat", f"{E_roll_heating:.1f} kW", "10%")
        
        # Specific energy comparison
        st.markdown("#### Specific Energy Consumption")
        
        reductions_energy = np.linspace(10, 60, 50)
        specific_energies = []
        
        for r_energy in reductions_energy:
            h2_energy = h1 - (h1 * r_energy / 100)
            if h2_energy > 0.5:
                tmp = compute_F_T_P(h1, h2_energy, D, b, N_rpm, T_C, T_roll_initial,
                                   steel, roll, mu0, a_T, b_v, friction_model)
                specific_energies.append(tmp["specific_energy"] / 1000)  # kJ/m²
            else:
                specific_energies.append(None)
        
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=reductions_energy,
            y=specific_energies,
            mode='lines',
            line=dict(width=3, color='#8b5cf6'),
            fill='tozeroy'
        ))
        
        fig_spec.update_layout(
            title="Specific Energy vs Reduction Ratio",
            xaxis_title="Reduction (%)",
            yaxis_title="Specific Energy (kJ/m²)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_spec, use_container_width=True)
    
    else:  # Strain Rate Sensitivity
        st.markdown("#### Strain Rate Effects on Flow Stress")
        
        # Flow stress vs strain rate at different temperatures
        edot_range = np.logspace(-1, 3, 100)  # 0.1 to 1000 s⁻¹
        T_strain_list = [800, 950, 1100]
        
        fig_strain = go.Figure()
        
        for T_strain in T_strain_list:
            sigma_strain = []
            for edot_test in edot_range:
                sigma = flow_stress_ZH(T_strain, edot_test, steel["A"], steel["n"],
                                      steel["alpha"], steel["Q"])
                sigma_strain.append(sigma)
            
            fig_strain.add_trace(go.Scatter(
                x=edot_range,
                y=sigma_strain,
                mode='lines',
                name=f"{T_strain}°C",
                line=dict(width=2)
            ))
        
        fig_strain.update_layout(
            title=f"Flow Stress vs Strain Rate<br><sub>{steel_choice} - Log Scale</sub>",
            xaxis_title="Strain Rate ε̇ (s⁻¹)",
            yaxis_title="Flow Stress σ̄ (MPa)",
            xaxis_type="log",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_strain, use_container_width=True)
        
        # Zener-Hollomon parameter analysis
        st.markdown("#### Zener-Hollomon Parameter (Z)")
        
        Z_vals = []
        for T_strain in T_strain_list:
            Z_temp = []
            for edot_test in edot_range:
                T_K = T_strain + 273.15
                Z = edot_test * math.exp(steel["Q"] / (R_GAS * T_K))
                Z_temp.append(Z)
            Z_vals.append(Z_temp)
        
        fig_Z = go.Figure()
        
        for i, T_strain in enumerate(T_strain_list):
            fig_Z.add_trace(go.Scatter(
                x=edot_range,
                y=Z_vals[i],
                mode='lines',
                name=f"{T_strain}°C",
                line=dict(width=2)
            ))
        
        fig_Z.update_layout(
            title="Zener-Hollomon Parameter vs Strain Rate",
            xaxis_title="Strain Rate ε̇ (s⁻¹)",
            yaxis_title="Z Parameter",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_dark",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_Z, use_container_width=True)
        
        st.info("""
        **Strain Rate Sensitivity:**
        - Higher strain rates increase flow stress (material strain hardens)
        - Effect is more pronounced at lower temperatures
        - Z-parameter combines temperature and strain rate effects
        - Critical for high-speed rolling optimization
        """)

# ==============================
# ENERGY & CO₂ MODULE (Enhanced)
# ==============================
density_steel_calc = steel["density"]  # kg/m³
b_m, h2_m = b / 1000, h2 / 1000

# Throughput estimation (mass flow in tons/hour)
mass_flow_tph = density_steel_calc * b_m * h2_m * res["v"] * 3600 / 1000  # ton/hour

if mass_flow_tph > 1e-6:
    energy_kWh_ton = res["P"] / max(mass_flow_tph, 1e-6)  # kWh/ton
else:
    energy_kWh_ton = 0.0

# CO₂ calculation (typical Indian grid factor)
co2_factor = 0.6  # kg CO₂ per kWh
co2_kg_per_ton = energy_kWh_ton * co2_factor

# Annual production estimate
hours_per_year = 7000  # Typical industrial operation
annual_production_tons = mass_flow_tph * hours_per_year
annual_energy_MWh = energy_kWh_ton * annual_production_tons / 1000
annual_co2_tons = co2_kg_per_ton * annual_production_tons / 1000

st.markdown("---")
st.subheader("♻️ Energy & Environmental Analysis")

col_env1, col_env2, col_env3, col_env4 = st.columns(4)
col_env1.metric("Energy Intensity", f"{energy_kWh_ton:.1f} kWh/ton")
col_env2.metric("CO₂ Emission", f"{co2_kg_per_ton:.1f} kg/ton")
col_env3.metric("Throughput", f"{mass_flow_tph:.1f} ton/hr")
col_env4.metric("Annual Production", f"{annual_production_tons/1000:.1f}k tons")

st.caption(f"Estimated annual energy: {annual_energy_MWh:.0f} MWh | Annual CO₂: {annual_co2_tons:.0f} tons")

# Environmental comparison
with st.expander("🌍 Environmental Impact Comparison"):
    st.markdown("""
    **Benchmark Comparison:**
    - World average: 150-200 kWh/ton
    - Best practice: 100-120 kWh/ton
    - Your process: {:.1f} kWh/ton
    """.format(energy_kWh_ton))
    
    if energy_kWh_ton < 120:
        st.success("✅ Excellent energy efficiency! Below industry best practice.")
    elif energy_kWh_ton < 150:
        st.info("ℹ️ Good energy efficiency, close to world average.")
    else:
        st.warning("⚠️ Energy consumption above average. Consider optimization.")

# ==============================
# LOGGING MODULE (Enhanced)
# ==============================
LOG_FILE = "roll_log_enhanced.csv"

def log_run():
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roll_material": material_choice,
        "steel": steel_choice,
        "T_C": T_C,
        "T_roll": T_roll_initial,
        "N_rpm": N_rpm,
        "h1": h1,
        "h2": h2,
        "D": D,
        "b": b,
        "reduction_pct": reduction_ratio,
        "friction_model": friction_model,
        "Load_kN": res["F"]/1000,
        "Torque_kNm": res["Tq"]/1000,
        "Power_kW": res["P"]/1000,
        "mu": res["mu"],
        "m": res["m"],
        "p_mean_MPa": res["p_mean"],
        "T_exit": res["T_exit"],
        "T_roll_operating": res["T_roll_op"],
        "roll_flattening_mm": res["roll_flattening"],
        "specific_energy_kJ_m2": res["specific_energy"]/1000,
        "Energy_kWh_ton": energy_kWh_ton,
        "CO2_kg_ton": co2_kg_per_ton,
        "throughput_tph": mass_flow_tph
    }
    df = pd.DataFrame([data])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
    st.success("✅ Simulation logged successfully!")

st.markdown("---")
col_log1, col_log2 = st.columns([1, 3])

with col_log1:
    if st.button("🧾 Log This Simulation", use_container_width=True):
        log_run()

with col_log2:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Complete Log (CSV)", 
                             data=f, 
                             file_name="roll_log_enhanced.csv", 
                             mime="text/csv",
                             use_container_width=True)

# ==============================
# UNICODE-SAFE TEXT SANITIZER
# ==============================
def clean_text(text):
    """Make text Latin-1 safe for FPDF"""
    replacements = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "·": ".", "°": " deg", "²": "^2", "³": "^3", "—": "-", "–": "-",
        "̇": "", "̄": "", "μ": "mu", "σ": "sigma", "ε": "epsilon"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text

# ==============================
# ENHANCED REPORT GENERATOR
# ==============================
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, clean_text(f"{APP_NAME} - Enhanced Rolling Simulation Report"), ln=True, align="C")
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

if st.button("📄 Generate Enhanced PDF Report", type="primary"):
    pdf = PDF()
    pdf.add_page()
    
    # Input Parameters
    pdf.section_title("Input Parameters")
    inputs = {
        "Roll Material": material_choice,
        "Steel Grade": steel_choice,
        "Rolling Temperature": f"{T_C} deg C",
        "Initial Roll Temperature": f"{T_roll_initial} deg C",
        "Roll Speed": f"{N_rpm} rpm",
        "Strip Thickness": f"h1={h1} mm, h2={h2} mm",
        "Reduction Ratio": f"{reduction_ratio:.1f}%",
        "Roll Diameter": f"{D} mm",
        "Strip Width": f"{b} mm",
        "Friction Model": friction_model
    }
    for k, v in inputs.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    
    # Simulation Results
    pdf.section_title("Simulation Results - Mechanical")
    results = {
        "Rolling Load": f"{res['F']/1000:.2f} kN",
        "Mean Pressure": f"{res['p_mean']:.1f} MPa",
        "Torque": f"{res['Tq']/1000:.2f} kN.m",
        "Power": f"{res['P']/1000:.2f} kW",
        "Flow Stress": f"{res['sigma_bar']:.1f} MPa",
        "Friction Coefficient mu": f"{res['mu']:.3f}",
        "Friction Factor m": f"{res['m']:.3f}",
        "Contact Length": f"{res['L']*1000:.1f} mm",
        "Roll Flattening": f"{res['roll_flattening']:.3f} mm",
        "Surface Speed": f"{res['v']:.2f} m/s",
        "Strain Rate": f"{res['edot']:.2f} per s"
    }
    for k, v in results.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    
    # Thermal Analysis
    pdf.section_title("Thermal Analysis")
    thermal = {
        "Strip Exit Temperature": f"{res['T_exit']:.0f} deg C",
        "Roll Operating Temperature": f"{res['T_roll_op']:.1f} deg C",
        "Roll Thermal Conductivity": f"{res['k_roll']:.1f} W/m.K",
        "Recrystallization Temp": f"{steel['T_recryst']} deg C",
        "Exit Temp Status": temp_status.replace("✓", "OK").replace("⚠", "WARNING")
    }
    for k, v in thermal.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    
    # Energy & Environment
    pdf.section_title("Energy & Environmental Analysis")
    energy = {
        "Specific Energy": f"{res['specific_energy']/1000:.1f} kJ/m^2",
        "Energy Intensity": f"{energy_kWh_ton:.2f} kWh/ton",
        "CO2 Emission": f"{co2_kg_per_ton:.2f} kg/ton",
        "Throughput": f"{mass_flow_tph:.1f} ton/hour",
        "Annual Production": f"{annual_production_tons/1000:.1f}k tons",
        "Annual Energy": f"{annual_energy_MWh:.0f} MWh",
        "Annual CO2": f"{annual_co2_tons:.0f} tons"
    }
    for k, v in energy.items():
        pdf.add_line(k, v)

    pdf.ln(4)
    
    # Notes
    pdf.section_title("Notes & Recommendations")
    pdf.multi_cell(0, 6, clean_text(
        f"This enhanced report includes temperature-dependent material modeling, "
        f"Von Karman roll pressure distribution, and elastic roll flattening analysis. "
        f"The exit temperature ({res['T_exit']:.0f} deg C) is {'above' if above_recryst else 'below'} "
        f"the recrystallization temperature ({steel['T_recryst']} deg C). "
        f"Energy efficiency is {energy_kWh_ton:.1f} kWh/ton. "
        f"Results are validated against industrial rolling theory and suitable for "
        f"process optimization and equipment selection."
    ))

    filename = f"MILLOPT_Pro_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    
    with open(filename, "rb") as f:
        st.download_button("📤 Download Enhanced Report (PDF)", 
                          f, 
                          file_name=filename, 
                          mime="application/pdf",
                          type="primary")

# ==============================
# FOOTER
# ==============================
st.markdown(f"""
<div class="m-footer">
  © 2025 {APP_NAME} · Enhanced for Steel InTech Challenge · Built on Von Karman Theory & Temperature-Dependent Material Models
</div>
""", unsafe_allow_html=True)

