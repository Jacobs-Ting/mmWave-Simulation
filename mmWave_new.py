import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import windows

st.set_page_config(page_title="3D Active mmWave TRX Array Simulator", layout="wide")
st.title("🛰️ 3D Active mmWave TRX Array Simulator")
st.write("Comprehensive 5G mmWave TDD System Simulator: Integrating RFIC Non-idealities, **5G NR Codebooks**, **Pattern Multiplication**, and **2D Taylor Window Tapering**.")

# ==========================================
# 1. Sidebar: Global Configuration
# ==========================================
st.sidebar.header("1. Array Geometry & Physical Settings")
Nx = st.sidebar.slider("Number of Elements (X-axis, Nx)", 1, 32, 8, step=1)
Ny = st.sidebar.slider("Number of Elements (Y-axis, Ny)", 1, 32, 8, step=1)
N_total = Nx * Ny

# --- Amplitude Tapering Settings ---
st.sidebar.subheader("📐 Amplitude Tapering (Windowing)")
tapering_mode = st.sidebar.radio("Tapering Strategy", ["Uniform (0 dB)", "Taylor Window"])
taylor_sll = 30.0
if tapering_mode == "Taylor Window":
    taylor_sll = st.sidebar.slider("Taylor Target SLL (dBc)", 15.0, 45.0, 30.0, step=1.0)

# Generate 2D Tapering Matrix
def get_2d_tapering(nx, ny, mode, sll):
    if mode == "Uniform (0 dB)":
        return np.ones((nx, ny)), 0.0
    else:
        # Generate 1D Taylor windows (nbar=4 is standard)
        w_x = windows.taylor(nx, nbar=4, sll=sll) if nx > 1 else np.array([1.0])
        w_y = windows.taylor(ny, nbar=4, sll=sll) if ny > 1 else np.array([1.0])
        # Outer product to get 2D matrix
        w_2d = np.outer(w_x, w_y)
        # Normalize weights to peak = 1.0
        w_2d = w_2d / np.max(w_2d)
        
        # Calculate Tapering Efficiency (Loss in dB)
        efficiency = (np.sum(w_2d)**2) / (N_total * np.sum(w_2d**2))
        taper_loss_db = 10 * np.log10(efficiency)
        return w_2d, taper_loss_db

taper_matrix, tapering_loss_db = get_2d_tapering(Nx, Ny, tapering_mode, taylor_sll)
st.sidebar.info(f"💡 Tapering Efficiency: **{tapering_loss_db:.2f} dB** loss")

st.sidebar.subheader("📐 Antenna Element Physics")
d_lambda = st.sidebar.slider("Element Spacing (d / λ)", 0.2, 1.0, 0.5, step=0.05)
element_type = st.sidebar.selectbox(
    "Antenna Element Type",
    ["Isotropic (Ideal Omni)", "Microstrip Patch (cos θ)", "High-Directivity Patch (cos² θ)"],
    index=1
)
elem_gain_dbi = st.sidebar.slider("Single Element Peak Gain (dBi)", 0.0, 10.0, 5.0, step=0.5)

st.sidebar.subheader("🔌 Passive Component Losses")
tr_switch_il_db = st.sidebar.slider("Front-end T/R Switch Loss (SPDT IL, dB)", 0.0, 5.0, 1.5, step=0.1)
ps_il_db = st.sidebar.slider("Phase Shifter Insertion Loss (PS IL, dB)", 0.0, 10.0, 4.0, step=0.5)
splitter_il_db = st.sidebar.slider("Per-stage 1-to-2 Splitter/Combiner Loss (dB)", 0.0, 3.0, 0.5, step=0.1)

# ==========================================
# 2. Beam Steering & 5G NR Codebook
# ==========================================
st.sidebar.header("2. Beam Steering Control")
steering_mode = st.sidebar.radio("Beam Generation Mode:", ["Continuous Physical Angle", "5G NR DFT Codebook (Type I)"])

kd = 2 * np.pi * d_lambda 
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
    c_o1, c_o2 = st.sidebar.columns(2)
    O_x = c_o1.selectbox("O1", [1, 2, 4], index=2)
    O_y = c_o2.selectbox("O2", [1, 2, 4], index=2)
    m_idx = st.sidebar.slider("Beam Index m", -Nx*O_x//2, Nx*O_x//2, 0)
    n_idx = st.sidebar.slider("Beam Index n", -Ny*O_y//2, Ny*O_y//2, 0)
    for ix in range(Nx):
        for iy in range(Ny):
            ideal_phase_matrix[ix, iy] = -2 * np.pi * ((ix * m_idx)/(Nx * O_x) + (iy * n_idx)/(Ny * O_y))
    u, v = 2 * m_idx / (Nx * O_x), 2 * n_idx / (Ny * O_y)
    sin_theta = np.sqrt(u**2 + v**2)
    if sin_theta <= 1.0:
        theta_0, phi_0 = np.degrees(np.arcsin(sin_theta)), np.degrees(np.arctan2(v, u))
    else:
        theta_0, phi_0 = 89.9, 0

st.sidebar.divider()

# ==========================================
# 3. RFIC Quantization & Physics Logic
# ==========================================
st.sidebar.subheader("⚙️ RFIC Phase Shifter Resolution")
ps_bits_option = st.sidebar.selectbox("PS Bits", ["Ideal", "6 bits", "5 bits", "3 bits"], index=2)
ps_bits = None if "Ideal" in ps_bits_option else int(ps_bits_option.split(" ")[0])

applied_phase_shifts = np.zeros((Nx, Ny)) 
phase_x_eval = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
phase_y_eval = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))

AF_target_val = 0j
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
        AF_target_val += taper_matrix[ix, iy] * np.exp(1j * (target_spatial_phase + actual_shift))

# Normalize by sum of weights for accurate gain calculation
AF_target_norm = np.abs(AF_target_val) / np.sum(taper_matrix)
quantization_loss_db = 20 * np.log10(AF_target_norm + 1e-12)

EF_target_mag = 1.0
if "cos²" in element_type: EF_target_mag = np.maximum(0, np.cos(np.radians(theta_0))**2)
elif "cos" in element_type: EF_target_mag = np.maximum(0, np.cos(np.radians(theta_0)))
scan_loss_db = -20 * np.log10(EF_target_mag + 1e-12) if theta_0 < 89 else 40.0

split_stages = np.log2(N_total) if N_total > 1 else 0
backend_passive_loss_db = (split_stages * splitter_il_db) + ps_il_db 
# Final Gain = N_Gain + Elem_Gain - Scan_Loss + Quant_Loss + Tapering_Loss
array_spatial_gain_dbi = 10 * np.log10(N_total) + elem_gain_dbi - scan_loss_db + quantization_loss_db + tapering_loss_db

# Shared 3D rendering computation block
@st.cache_data
def compute_3d_pattern(Nx, Ny, kd, applied_phase_shifts, N_total, element_type, taper_matrix):
    # --- HIGHER RESOLUTION FOR PENCIL BEAMS ---
    theta = np.linspace(0, np.pi/2, 200)
    phi = np.linspace(-np.pi, np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    AF = np.zeros_like(THETA, dtype=complex)
    for ix in range(Nx):
        for iy in range(Ny):
            spatial_phase = ix * kd * np.sin(THETA)*np.cos(PHI) + iy * kd * np.sin(THETA)*np.sin(PHI)
            AF += taper_matrix[ix, iy] * np.exp(1j * (spatial_phase + applied_phase_shifts[ix, iy]))
    AF_norm = np.abs(AF) / np.sum(taper_matrix)
    EF = np.cos(THETA)**2 if "cos²" in element_type else (np.cos(THETA) if "cos" in element_type else np.ones_like(THETA))
    Total_Pattern_Linear = AF_norm * EF
    R = Total_Pattern_Linear
    return R * np.sin(THETA) * np.cos(PHI), R * np.sin(THETA) * np.sin(PHI), R * np.cos(THETA), Total_Pattern_Linear

X_3d, Y_3d, Z_3d, Pattern_Linear_3d = compute_3d_pattern(Nx, Ny, kd, applied_phase_shifts, N_total, element_type, taper_matrix)

# ==========================================
# 4. Main UI Layout
# ==========================================
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Link Direction:", ["Tx Mode (Transmit)", "Rx Mode (Receive)"])

if mode == "Tx Mode (Transmit)":
    st.subheader("📊 Tx Link Budget (with 2D Tapering Compensation)")
    pa_arch = st.sidebar.radio("PA Architecture:", ["Distributed PA", "Centralized PA"])
    feed_power_dbm = st.sidebar.number_input("System Input Power (dBm)", value=0.0)
    pa_gain_db = st.sidebar.number_input("PA Gain (dB)", value=15.0)
    
    if pa_arch == "Distributed PA":
        power_before_pa = feed_power_dbm - (10 * np.log10(N_total) + backend_passive_loss_db)
        pout_per_pa_dbm = power_before_pa + pa_gain_db
        power_at_antenna_dbm = pout_per_pa_dbm - tr_switch_il_db
        pa_display_label = "Per-PA Pout (分散式)"
        pa_display_val = pout_per_pa_dbm
    else:
        pout_central_pa_dbm = feed_power_dbm + pa_gain_db
        power_at_antenna_dbm = pout_central_pa_dbm - (10 * np.log10(N_total) + backend_passive_loss_db) - tr_switch_il_db
        pa_display_label = "Central PA Pout (集中式)"
        pa_display_val = pout_central_pa_dbm
        
    total_conducted_tx_dbm = power_at_antenna_dbm + 10 * np.log10(N_total)
    eirp_dbm = total_conducted_tx_dbm + array_spatial_gain_dbi
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(pa_display_label, f"{pa_display_val:.1f} dBm")
    c2.metric("Tapering Efficiency", f"{tapering_loss_db:.2f} dB", delta_color="inverse")
    c3.metric("Array Gain (dBi)", f"{array_spatial_gain_dbi:.2f}")
    c4.metric("Target EIRP", f"{eirp_dbm:.1f} dBm")
    
elif mode == "Rx Mode (Receive)":
    st.subheader("🎧 Rx Noise & Sensitivity Analysis (with 2D Tapering Compensation)")
    rx_arch = st.sidebar.radio("LNA Architecture:", ["Distributed LNA", "Centralized LNA"])
    
    col_rx1, col_rx2, col_rx3 = st.columns(3)
    bw_mhz = col_rx1.number_input("Channel Bandwidth (MHz)", value=100.0, step=10.0, min_value=1.0)
    lna_gain_db = col_rx2.number_input("LNA Gain (dB)", value=20.0, step=1.0)
    lna_nf_db = col_rx3.number_input("LNA Noise Figure (dB)", value=3.0, step=0.1)
    snr_min_db = st.sidebar.number_input("Required System SNR (dB)", value=1.5, step=0.5)

    f_lna = 10 ** (lna_nf_db / 10)
    g_lna_lin = 10 ** (lna_gain_db / 10)
    f_switch = 10 ** (tr_switch_il_db / 10)
    g_switch_lin = 10 ** (-tr_switch_il_db / 10) 
    f_passive = 10 ** (backend_passive_loss_db / 10)
    
    if rx_arch == "Distributed LNA":
        f_cascaded = f_switch + (f_lna - 1) / g_switch_lin + (f_passive - 1) / (g_switch_lin * g_lna_lin)
    else:
        total_front_loss_db = tr_switch_il_db + backend_passive_loss_db
        f_front_passive = 10 ** (total_front_loss_db / 10)
        g_front_passive = 10 ** (-total_front_loss_db / 10)
        f_cascaded = f_front_passive + (f_lna - 1) / g_front_passive

    nf_cascaded_db = 10 * np.log10(f_cascaded)
    
    kTB_dbm = -174 + 10 * np.log10(bw_mhz * 1e6)
    conducted_sens_dbm = kTB_dbm + nf_cascaded_db + snr_min_db
    target_eis_dbm = conducted_sens_dbm - array_spatial_gain_dbi

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Thermal Noise (kTB)", f"{kTB_dbm:.1f} dBm")
    c2.metric("Cascaded System NF", f"{nf_cascaded_db:.2f} dB")
    c3.metric("Conducted Sensitivity", f"{conducted_sens_dbm:.2f} dBm")
    
    taper_penalty_str = f"{-tapering_loss_db:.2f} dB Taper Penalty" if tapering_loss_db < 0 else "0.00 dB"
    c4.metric("Target EIS", f"{target_eis_dbm:.2f} dBm", delta=taper_penalty_str, delta_color="inverse")

st.divider()

# ==========================================
# 5. Rendering 3D and 2D Plots Side-by-Side
# ==========================================

baseline_gain = 10 * np.log10(N_total) + elem_gain_dbi + quantization_loss_db + tapering_loss_db

# --- 3D Plot Generation ---
Gain_Matrix_dBi = baseline_gain + 20 * np.log10(Pattern_Linear_3d + 1e-12)
fig_3d = go.Figure(data=[go.Surface(x=X_3d, y=Y_3d, z=Z_3d, surfacecolor=Gain_Matrix_dBi, colorscale='Jet', cmax=baseline_gain, cmin=baseline_gain-40)])

title_3d = "3D Radiation Pattern - Taylor Window Enabled" if tapering_mode == "Taylor Window" else "3D Radiation Pattern - Uniform Array"
fig_3d.update_layout(
    title=title_3d, 
    scene=dict(
        xaxis=dict(title='X', range=[-1, 1], autorange=False, showbackground=False),
        yaxis=dict(title='Y', range=[-1, 1], autorange=False, showbackground=False),
        zaxis=dict(title='Z (Broadside)', range=[0, 1], autorange=False, showbackground=False),
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
)

# --- NEW: 2D Elevation Cut Computation ---
# We calculate a precise 1D array factor sweep along the target Azimuth plane (phi = phi_0)
phi_cut_rad = np.radians(phi_0)
theta_1d = np.linspace(-np.pi/2, np.pi/2, 400)
AF_1d = np.zeros_like(theta_1d, dtype=complex)

for ix in range(Nx):
    for iy in range(Ny):
        # spatial phase sweeping across Theta from -90 to +90 degrees at the target Phi
        sp = ix * kd * np.sin(theta_1d)*np.cos(phi_cut_rad) + iy * kd * np.sin(theta_1d)*np.sin(phi_cut_rad)
        AF_1d += taper_matrix[ix, iy] * np.exp(1j * (sp + applied_phase_shifts[ix, iy]))

AF_1d_norm = np.abs(AF_1d) / np.sum(taper_matrix)

# 1D Element Factor
if "cos²" in element_type:
    EF_1d = np.cos(theta_1d)**2
elif "cos" in element_type:
    EF_1d = np.cos(theta_1d)
else:
    EF_1d = np.ones_like(theta_1d)

Pattern_1d_Linear = AF_1d_norm * np.maximum(0, EF_1d)
Gain_1d_dBi = baseline_gain + 20 * np.log10(Pattern_1d_Linear + 1e-12)

# --- NEW: 2D Plot Generation ---
fig_2d = go.Figure()
fig_2d.add_trace(go.Scatter(
    x=np.degrees(theta_1d), y=Gain_1d_dBi, mode='lines', 
    line=dict(color='#00e5ff', width=3), 
    fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)',
    name=f"Azimuth φ={phi_0:.1f}°"
))

fig_2d.update_layout(
    title=f"2D Principal Elevation Cut (Azimuth φ = {phi_0:.0f}°)",
    xaxis_title="Scanning Angle θ (degrees)",
    yaxis_title="Gain (dBi)",
    yaxis=dict(range=[baseline_gain - 40, baseline_gain + 5]),
    xaxis=dict(range=[-90, 90], tick0=-90, dtick=30),
    margin=dict(l=0, r=0, b=40, t=40),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
)

# --- Display Side-by-Side ---
col_plot1, col_plot2 = st.columns(2)
with col_plot1:
    st.plotly_chart(fig_3d, use_container_width=True)
with col_plot2:
    st.plotly_chart(fig_2d, use_container_width=True)