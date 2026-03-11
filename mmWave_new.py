import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D 主動式 mmWave TRX 陣列模擬器", layout="wide")
st.title("🛰️ 3D 主動式 mmWave TRX 陣列模擬器")
st.write("完整對應 5G 毫米波 TDD 系統：整合真實 RFIC 物理特性 (前端損耗、移相器量化) 與 **5G NR Baseband 波束碼本 (Beam Codebook)**。")

# ==========================================
# 1. 側邊欄：全域共用參數 (Shared Parameters)
# ==========================================
st.sidebar.header("1. 陣列幾何與共用物理設定")
Nx = st.sidebar.slider("X 軸天線數量 (Nx)", 1, 32, 8, step=1)
Ny = st.sidebar.slider("Y 軸天線數量 (Ny)", 1, 32, 8, step=1)
N_total = Nx * Ny

elem_gain_dbi = st.sidebar.slider("單一天線 Peak Gain (dBi)", 0.0, 10.0, 5.0, step=0.5)

st.sidebar.subheader("🔌 實體被動元件損耗 (Passive Losses)")
tr_switch_il_db = st.sidebar.slider("前端 T/R 開關損耗 (SPDT IL, dB)", 0.0, 5.0, 1.5, step=0.1)
ps_il_db = st.sidebar.slider("移相器 插入損耗 (Phase Shifter IL, dB)", 0.0, 10.0, 4.0, step=0.5)
splitter_il_db = st.sidebar.slider("單級 1-to-2 分配/合成器 額外損耗 (dB)", 0.0, 3.0, 0.5, step=0.1)

# ==========================================
# [全新功能] 2. 5G NR 波束碼本與指向控制
# ==========================================
st.sidebar.header("2. 波束指向控制 (Beam Steering)")
steering_mode = st.sidebar.radio("波束生成模式 (Baseband Control)：", 
                                 ["自由物理角度 (Continuous)", "5G NR DFT Codebook (Type I)"])

kd = np.pi # d=lambda/2
ideal_phase_matrix = np.zeros((Nx, Ny))

if steering_mode == "自由物理角度 (Continuous)":
    theta_0 = st.sidebar.slider("目標仰角 (Zenith, 0~90度)", 0, 90, 30, step=1)
    phi_0 = st.sidebar.slider("目標方位角 (Azimuth, -180~180度)", -180, 180, 45, step=5)
    
    phase_x_target = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
    phase_y_target = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))
    
    for ix in range(Nx):
        for iy in range(Ny):
            ideal_phase_matrix[ix, iy] = -(ix * kd * phase_x_target + iy * kd * phase_y_target)

elif steering_mode == "5G NR DFT Codebook (Type I)":
    st.sidebar.markdown("*(對應 3GPP TS 38.214 - 2D DFT 過取樣波束矩陣)*")
    c_o1, c_o2 = st.sidebar.columns(2)
    O_x = c_o1.selectbox("X軸過取樣 (O1)", [1, 2, 4], index=2, help="Oversampling factor (標準通常為4)")
    O_y = c_o2.selectbox("Y軸過取樣 (O2)", [1, 2, 4], index=2)
    
    # 建立以 0 為中心的 Index，方便直覺對應物理角度
    m_idx = st.sidebar.slider("Beam Index m (X軸/方位角)", -Nx*O_x//2, Nx*O_x//2, 0)
    n_idx = st.sidebar.slider("Beam Index n (Y軸/仰角)", -Ny*O_y//2, Ny*O_y//2, 0)
    
    # 計算 3GPP 定義的 DFT 相位差
    for ix in range(Nx):
        for iy in range(Ny):
            ideal_phase_matrix[ix, iy] = -2 * np.pi * ((ix * m_idx)/(Nx * O_x) + (iy * n_idx)/(Ny * O_y))
            
    # 將 Codebook 索引反推回真實世界的物理角度 (用來計算 Scan Loss 與顯示)
    u = 2 * m_idx / (Nx * O_x)
    v = 2 * n_idx / (Ny * O_y)
    sin_theta = np.sqrt(u**2 + v**2)
    
    if sin_theta <= 1.0:
        theta_0 = np.degrees(np.arcsin(sin_theta))
        phi_0 = np.degrees(np.arctan2(v, u))
        st.sidebar.success(f"📌 此 Codebook 對應物理角度：\n仰角 {theta_0:.1f}°, 方位角 {phi_0:.1f}°")
    else:
        theta_0 = 89.9
        phi_0 = 0
        st.sidebar.error("⚠️ 此 Index 組合落在『不可見區域 (Invisible Region)』，無法形成有效波束！")

st.sidebar.divider()

# ==========================================
# 3. 核心物理運算：套用 RFIC 相位量化誤差
# ==========================================
st.sidebar.subheader("⚙️ RFIC 移相器解析度")
ps_bits_option = st.sidebar.selectbox(
    "選擇移相器位元數 (Phase Quantization)",
    ["Ideal (無窮解析度)", "6 bits (64階, 步階 5.6°)", "5 bits (32階, 步階 11.25°)", "3 bits (8階, 步階 45°)"],
    index=2
)

ps_bits = None if "Ideal" in ps_bits_option else int(ps_bits_option.split(" ")[0])

AF_target_val = 0j
applied_phase_shifts = np.zeros((Nx, Ny)) 

# 為了計算量化損耗，我們需要知道如果打在這個角度，理想上的完美空間相位差是多少
phase_x_eval = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
phase_y_eval = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))

for ix in range(Nx):
    for iy in range(Ny):
        ideal_shift = ideal_phase_matrix[ix, iy]
        
        # RFIC 晶片的殘酷：只能選擇最近的檔位 (量化)
        if ps_bits is not None:
            step = 2 * np.pi / (2**ps_bits)
            actual_shift = np.round(ideal_shift / step) * step
        else:
            actual_shift = ideal_shift
            
        applied_phase_shifts[ix, iy] = actual_shift
        
        target_spatial_phase = ix * kd * phase_x_eval + iy * kd * phase_y_eval
        AF_target_val += np.exp(1j * (target_spatial_phase + actual_shift))

# 量化損耗計算 (Quantization Loss)
AF_target_norm = np.abs(AF_target_val) / N_total
quantization_loss_db = 20 * np.log10(AF_target_norm + 1e-12)

# [修正 Bug] 單天線無 Scan Loss 
if N_total > 1:
    scan_loss_db = -10 * np.log10(np.cos(np.radians(theta_0))) if theta_0 < 89 else 20.0
else:
    scan_loss_db = 0.0

split_stages = np.log2(N_total) if N_total > 1 else 0
backend_passive_loss_db = (split_stages * splitter_il_db) + ps_il_db 
array_spatial_gain_dbi = 10 * np.log10(N_total) + elem_gain_dbi - scan_loss_db + quantization_loss_db 

# ==========================================
# 4. 側邊欄：Tx / Rx 模式切換與專屬參數
# ==========================================
st.sidebar.header("3. 工作模式切換 (Operation Mode)")
mode = st.sidebar.radio("選擇鏈路方向：", ["Tx 模式 (發射鏈路)", "Rx 模式 (接收鏈路)"])

if mode == "Tx 模式 (發射鏈路)":
    feed_power_dbm = 0.0
    pa_gain_db = 15.0
    rad_eff_db = -2.0
    min_eirp_dbm = 22.4
    max_trp_dbm = 23.0

    pin_pa_dbm = feed_power_dbm - (10 * np.log10(N_total) + backend_passive_loss_db)
    pout_pa_dbm = pin_pa_dbm + pa_gain_db
    power_at_antenna_element_dbm = pout_pa_dbm - tr_switch_il_db
    total_conducted_tx_dbm = power_at_antenna_element_dbm + 10 * np.log10(N_total)
    
    trp_dbm = total_conducted_tx_dbm + rad_eff_db
    eirp_dbm = total_conducted_tx_dbm + array_spatial_gain_dbi

    st.subheader("📊 Tx 鏈路預算與 3GPP 規範驗證")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("單一 PA 輸出功率", f"{pout_pa_dbm:.1f} dBm")
    c2.metric("抵達單一天線之功率", f"{power_at_antenna_element_dbm:.1f} dBm")
    c3.metric("陣列空間總增益", f"{array_spatial_gain_dbi:.2f} dBi", delta=f"波束量化損耗 {quantization_loss_db:.2f} dB", delta_color="normal")
    c4.metric("陣列真實導通功率", f"{total_conducted_tx_dbm:.1f} dBm")

    st.markdown("**(3) 🎯 3GPP Class 3 規格驗證**")
    c5, c6 = st.columns(2)
    c5.metric("📦 預估 TRP", f"{trp_dbm:.1f} dBm", delta=f"{trp_dbm - max_trp_dbm:+.1f} dB (vs Max TRP)")
    c6.metric("🚀 預估 Peak EIRP", f"{eirp_dbm:.1f} dBm", delta=f"{eirp_dbm - min_eirp_dbm:+.1f} dB (vs Min EIRP)", delta_color="normal")

elif mode == "Rx 模式 (接收鏈路)":
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

    st.subheader("🎧 Rx 系統雜訊與靈敏度物理推導")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. 系統熱雜訊", f"{kTB_dbm:.1f} dBm")
    c2.metric("2. 前端 T/R 損耗", f"{tr_switch_il_db:.1f} dB")
    c3.metric("3. 整機串聯 NF", f"{nf_cascaded_db:.2f} dB")
    c4.metric("4. 傳導靈敏度", f"{conducted_sens_dbm:.2f} dBm")

    st.markdown("**(2) 空間輻射層面 (OTA & 3GPP 驗證)**")
    c5, c6 = st.columns(2)
    c5.metric("📡 Rx 陣列空間總增益", f"+{array_spatial_gain_dbi:.2f} dBi", delta=f"波束量化損耗 {quantization_loss_db:.2f} dB", delta_color="normal")
    c6.metric("🎯 預估 Peak EIS", f"{peak_eis_dbm:.2f} dBm", delta=f"{peak_eis_dbm - target_eis_dbm:+.2f} dB (vs 目標 EIS)", delta_color="inverse")

st.divider()

# ==========================================
# 5. Plotly 3D 視覺化 (真實物理場型：套用量化相位)
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

title_str = f"3D Radiation Pattern (Target: θ={theta_0:.1f}°, φ={phi_0:.1f}°) - "
title_str += "Baseband Codebook" if steering_mode != "自由物理角度 (Continuous)" else "Custom Angle"

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
        zaxis=dict(title='Z (正前方)', range=[0, 1], showbackground=False),
        aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    width=800, height=600, margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig, use_container_width=True)