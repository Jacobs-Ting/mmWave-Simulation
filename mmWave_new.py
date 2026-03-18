import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Active mmWave TRX Array Simulator", layout="wide")
st.title("🛰️ 3D Active mmWave TRX Array Simulator")
st.write("Comprehensive 5G mmWave TDD System Simulator: Integrating RFIC Non-idealities (T/R Loss, Phase Quantization) and **5G NR Baseband Beam Codebooks**.")

# ==========================================
# 1. Sidebar: Global Configuration
# ==========================================
st.sidebar.header("1. Array Geometry & Physical Settings")
Nx = st.sidebar.slider("Number of Elements (X-axis, Nx)", 1, 32, 8, step=1)
Ny = st.sidebar.slider("Number of Elements (Y-axis, Ny)", 1, 32, 8, step=1)
N_total = Nx * Ny
st.sidebar.info(f"💡 Total Elements: **{N_total}** (Corresponding to {N_total} TRX Channels)")

elem_gain_dbi = st.sidebar.slider("Single Element Peak Gain (dBi)", 0.0, 10.0, 5.0, step=0.5)

st.sidebar.subheader("🔌 Passive Component Losses")
tr_switch_il_db = st.sidebar.slider("Front-end T/R Switch Loss (SPDT IL, dB)", 0.0, 5.0, 1.5, step=0.1)
ps_il_db = st.sidebar.slider("Phase Shifter Insertion Loss (PS IL, dB)", 0.0, 10.0, 4.0, step=0.5)
splitter_il_db = st.sidebar.slider("Per-stage 1-to-2 Splitter/Combiner Loss (dB)", 0.0, 3.0, 0.5, step=0.1)

# ==========================================
# 2. Beam Steering & 5G NR Codebook
# ==========================================
st.sidebar.header("2. Beam Steering Control")
steering_mode = st.sidebar.radio("Beam Generation Mode (Baseband Control):", 
                                 ["Continuous Physical Angle", "5G NR DFT Codebook (Type I)"])

kd = np.pi # d=lambda/2
ideal_phase_matrix = np.zeros((Nx, Ny))

if steering_mode == "Continuous Physical Angle":
    theta_0 = st.sidebar.slider("Target Zenith Angle (θ, 0~90°)", 0, 90, 30, step=1)
    phi_0 = st.sidebar.slider("Target Azimuth Angle (φ, -180~180°)", -180, 180, 45, step=5)
    
    phase_x_target = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
    phase_y_target = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))
    
    for ix in range(Nx):
        for iy in range(Ny):
            ideal_phase_matrix[ix, iy] = -(ix * kd * phase_x_target + iy * kd * phase_y_target)

elif steering_mode == "5G NR DFT Codebook (Type I)":
    st.sidebar.markdown("*(Ref: 3GPP TS 38.214 - 2D DFT Oversampled Matrix)*")
    c_o1, c_o2 = st.sidebar.columns(2)
    O_x = c_o1.selectbox("X-axis Oversampling (O1)", [1, 2, 4], index=2)
    O_y = c_o2.selectbox("Y-axis Oversampling (O2)", [1, 2, 4], index=2)
    
    m_idx = st.sidebar.slider("Beam Index m (X-axis)", -Nx*O_x//2, Nx*O_x//2, 0)
    n_idx = st.sidebar.slider("Beam Index n (Y-axis)", -Ny*O_y//2, Ny*O_y//2, 0)
    
    for ix in range(Nx):
        for iy in range(Ny):
            ideal_phase_matrix[ix, iy] = -2 * np.pi * ((ix * m_idx)/(Nx * O_x) + (iy * n_idx)/(Ny * O_y))
            
    u = 2 * m_idx / (Nx * O_x)
    v = 2 * n_idx / (Ny * O_y)
    sin_theta = np.sqrt(u**2 + v**2)
    
    if sin_theta <= 1.0:
        theta_0 = np.degrees(np.arcsin(sin_theta))
        phi_0 = np.degrees(np.arctan2(v, u))
        st.sidebar.success(f"📌 Equivalent Physical Angle:\nZenith {theta_0:.1f}°, Azimuth {phi_0:.1f}°")
    else:
        theta_0 = 89.9
        phi_0 = 0
        st.sidebar.error("⚠️ Invisible Region: Index combination results in evanescent waves!")

st.sidebar.divider()

# ==========================================
# 3. RFIC Quantization Logic
# ==========================================
st.sidebar.subheader("⚙️ RFIC Phase Shifter Resolution")
ps_bits_option = st.sidebar.selectbox(
    "Select PS Bits (Phase Quantization)",
    ["Ideal (Infinite Resolution)", "6 bits (64 steps, 5.6°)", "5 bits (32 steps, 11.25°)", "3 bits (8 steps, 45°)"],
    index=2
)

ps_bits = None if "Ideal" in ps_bits_option else int(ps_bits_option.split(" ")[0])

AF_target_val = 0j
applied_phase_shifts = np.zeros((Nx, Ny)) 

phase_x_eval = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
phase_y_eval = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))

for ix in range(Nx):
    for iy in range(Ny):
        ideal_shift = ideal_phase_matrix[ix, iy]
        
        if ps_bits is not None:
            step = 2 * np.pi / (2**ps_bits)
            actual_shift = np.round(ideal_shift / step) * step
        else:
            actual_shift = ideal_shift
            
        applied_phase_shifts[ix, iy] = actual_shift
        
        target_spatial_phase = ix * kd * phase_x_eval + iy * kd * phase_y_eval
        AF_target_val += np.exp(1j * (target_spatial_phase + actual_shift))

AF_target_norm = np.abs(AF_target_val) / N_total
quantization_loss_db = 20 * np.log10(AF_target_norm + 1e-12)

if N_total > 1:
    scan_loss_db = -10 * np.log10(np.cos(np.radians(theta_0))) if theta_0 < 89 else 20.0
else:
    scan_loss_db = 0.0

split_stages = np.log2(N_total) if N_total > 1 else 0
backend_passive_loss_db = (split_stages * splitter_il_db) + ps_il_db 
array_spatial_gain_dbi = 10 * np.log10(N_total) + elem_gain_dbi - scan_loss_db + quantization_loss_db 

# ==========================================
# 4. Mode Selection: Tx / Rx Link Budget
# ==========================================
st.sidebar.header("3. Operation Mode Selection")
mode = st.sidebar.radio("Select Link Direction:", ["Tx Mode (Transmit)", "Rx Mode (Receive)"])

if mode == "Tx Mode (Transmit)":
    st.sidebar.subheader("📡 Tx RF & PA Settings")
    feed_power_dbm = st.sidebar.number_input("System Input Power (dBm)", value=0.0, step=1.0)
    pa_gain_db = st.sidebar.number_input("Single PA Gain (dB)", value=15.0, step=1.0)
    rad_eff_db = st.sidebar.slider("Antenna Radiation Efficiency (dB)", -5.0, 0.0, -2.0, step=0.5)
    
    st.sidebar.subheader("🎯 3GPP Tx Specifications")
    min_eirp_dbm = st.sidebar.number_input("3GPP Min Peak EIRP (dBm)", value=22.4, step=0.1)
    max_trp_dbm = st.sidebar.number_input("3GPP Max TRP (dBm)", value=23.0, step=0.1)

    # Core Calculations
    pin_pa_dbm = feed_power_dbm - (10 * np.log10(N_total) + backend_passive_loss_db)
    pout_pa_dbm = pin_pa_dbm + pa_gain_db
    power_at_antenna_element_dbm = pout_pa_dbm - tr_switch_il_db
    total_conducted_tx_dbm = power_at_antenna_element_dbm + 10 * np.log10(N_total)
    
    trp_dbm = total_conducted_tx_dbm + rad_eff_db
    eirp_dbm = total_conducted_tx_dbm + array_spatial_gain_dbi

    # --- TOP DATA PANEL ---
    st.subheader("📊 Tx Link Budget & 3GPP Compliance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Per-PA Pout", f"{pout_pa_dbm:.1f} dBm")
    c2.metric("Power at Antenna", f"{power_at_antenna_element_dbm:.1f} dBm")
    c3.metric("Array Spatial Gain", f"{array_spatial_gain_dbi:.2f} dBi", 
              delta=f"Quantization Loss {quantization_loss_db:.2f} dB", delta_color="normal")
    c4.metric("Total Conducted Tx", f"{total_conducted_tx_dbm:.1f} dBm")

    # --- BOTTOM DATA PANEL (The Re-added Section) ---
    st.write("") 
    st.markdown("**(Regulatory) 🎯 3GPP Class 3 Compliance Verification**")
    c5, c6 = st.columns(2)
    with c5:
        trp_delta = trp_dbm - max_trp_dbm
        st.metric("📦 Estimated TRP", f"{trp_dbm:.1f} dBm", 
                  delta=f"{trp_delta:+.1f} dB (vs Max TRP)", delta_color="inverse")
    with c6:
        eirp_delta = eirp_dbm - min_eirp_dbm
        st.metric("🚀 Estimated Peak EIRP", f"{eirp_dbm:.1f} dBm", 
                  delta=f"{eirp_delta:+.1f} dB (vs Min EIRP)", delta_color="normal"))
    
elif mode == "Rx Mode (Receive)":
    st.subheader("🎧 Rx Noise & Sensitivity Analysis")
    lna_gain_db = 20.0
    lna_nf_db = 3.0
    bw_mhz = 100.0
    snr_min_db = -1.0 + 2.5
    target_eis_dbm = -88.0

    f_switch = 10 ** (tr_switch_il_db / 10)
    g_switch_lin = 10 ** (-tr_switch_il_db / 10) 
    f_lna = 10 ** (lna_nf_db / 10)
    g_lna_lin = 10 ** (lna_gain_db / 10)
    f_passive = 10 ** (backend_passive_loss_db / 10)
    
    f_cascaded = f_switch + (f_lna - 1) / g_switch_lin + (f_passive - 1) / (g_switch_lin * g_lna_lin)
    nf_cascaded_db = 10 * np.log10(f_cascaded)
    kTB_dbm = -174 + 10 * np.log10(bw_mhz * 1e6)
    conducted_sens_dbm = kTB_dbm + nf_cascaded_db + snr_min_db
    peak_eis_dbm = conducted_sens_dbm - array_spatial_gain_dbi

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Thermal Noise (kTB)", f"{kTB_dbm:.1f} dBm")
    c2.metric("2. T/R Switch IL", f"{tr_switch_il_db:.1f} dB")
    c3.metric("3. System Cascaded NF", f"{nf_cascaded_db:.2f} dB")
    c4.metric("4. Conducted Sensitivity", f"{conducted_sens_dbm:.2f} dBm")

    st.markdown("**(OTA) 🛰️ Space Propagation & 3GPP Compliance**")
    c5, c6 = st.columns(2)
    c5.metric("📡 Rx Spatial Gain", f"+{array_spatial_gain_dbi:.2f} dBi", delta=f"Quantization Loss {quantization_loss_db:.2f} dB", delta_color="normal")
    c6.metric("🎯 Estimated Peak EIS", f"{peak_eis_dbm:.2f} dBm", delta=f"{peak_eis_dbm - target_eis_dbm:+.2f} dB (vs Target EIS)", delta_color="inverse")

st.divider()

# ==========================================
# 5. Plotly 3D Visualization
# ==========================================
theta = np.linspace(0, np.pi/2, 60)
phi = np.linspace(-np.pi, np.pi, 120)
THETA, PHI = np.meshgrid(theta, phi)

AF = np.zeros_like(THETA, dtype=complex)

for ix in range(Nx):
    for iy in range(Ny):
        spatial_phase = ix * kd * np.sin(THETA)*np.cos(PHI) + iy * kd * np.sin(THETA)*np.sin(PHI)
        AF += np.exp(1j * (spatial_phase + applied_phase_shifts[ix, iy]))
        
AF_norm = np.abs(AF) / N_total
EF = np.cos(THETA)
Total_Pattern_Linear = AF_norm * EF

R = Total_Pattern_Linear
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

Gain_Matrix_dBi = array_spatial_gain_dbi + 20 * np.log10(Total_Pattern_Linear + 1e-12)
cmin_val = np.max([array_spatial_gain_dbi - 40, -20])

title_str = f"3D Radiation Pattern (Target: θ={theta_0:.1f}°, φ={phi_0:.1f}°) - Real-world RFIC Mode"

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z, surfacecolor=Gain_Matrix_dBi, 
    colorscale='Jet', cmin=cmin_val, cmax=array_spatial_gain_dbi,
    colorbar=dict(title="Gain (dBi)", thickness=20)
)])

fig.update_layout(
    title=title_str,
    scene=dict(
        xaxis=dict(title='X', range=[-1, 1], showbackground=False),
        yaxis=dict(title='Y', range=[-1, 1], showbackground=False),
        zaxis=dict(title='Z (Broadside)', range=[0, 1], showbackground=False),
        aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    width=800, height=600, margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig, use_container_width=True)
