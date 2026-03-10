import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D 主動式 mmWave TRX 陣列模擬器", layout="wide")
st.title("🛰️ 3D 主動式 mmWave TRX 陣列模擬器")
st.write("完整對應 5G 毫米波 TDD 收發機系統：整合真實 RFIC 物理特性，包含 **前端 T/R 開關損耗** 與 **移相器量化誤差 (Phase Quantization)**。")

# ==========================================
# 1. 側邊欄：全域共用參數 (Shared Parameters)
# ==========================================
st.sidebar.header("1. 陣列幾何與共用物理設定")
Nx = st.sidebar.slider("X 軸天線數量 (Nx)", 1, 32, 8, step=1)
Ny = st.sidebar.slider("Y 軸天線數量 (Ny)", 1, 32, 8, step=1)
N_total = Nx * Ny
st.sidebar.info(f"💡 總天線數：**{N_total}** 根 (對應 {N_total} 組 TRX 通道)")

elem_gain_dbi = st.sidebar.slider("單一天線 Peak Gain (dBi)", 0.0, 10.0, 5.0, step=0.5)

st.sidebar.subheader("🔌 實體被動元件損耗 (Passive Losses)")
tr_switch_il_db = st.sidebar.slider("前端 T/R 開關損耗 (SPDT IL, dB)", 0.0, 5.0, 1.5, step=0.1)
ps_il_db = st.sidebar.slider("移相器 插入損耗 (Phase Shifter IL, dB)", 0.0, 10.0, 4.0, step=0.5)
splitter_il_db = st.sidebar.slider("單級 1-to-2 分配/合成器 額外損耗 (dB)", 0.0, 3.0, 0.5, step=0.1)

st.sidebar.header("2. 波束指向控制 (Beam Steering)")
theta_0 = st.sidebar.slider("目標仰角 (Zenith, 0~90度)", 0, 90, 30, step=1)
phi_0 = st.sidebar.slider("目標方位角 (Azimuth, -180~180度)", -180, 180, 45, step=5)

# [新增] 移相器解析度 (量化誤差來源)
st.sidebar.subheader("⚙️ RFIC 移相器解析度")
ps_bits_option = st.sidebar.selectbox(
    "選擇移相器位元數 (Phase Quantization)",
    ["Ideal (無窮解析度)", "6 bits (64階, 步階 5.6°)", "5 bits (32階, 步階 11.25°)", "3 bits (8階, 步階 45°)"],
    index=2, help="真實晶片無法提供完美相位，只能逼近最接近的檔位。這會導致增益下降與旁瓣升高。"
)

# 解析位元數
if "Ideal" in ps_bits_option:
    ps_bits = None
else:
    ps_bits = int(ps_bits_option.split(" ")[0])

st.sidebar.divider()

# ==========================================
# 3. 核心物理運算：陣列空間相位與量化誤差 (Pre-calculation)
# ==========================================
kd = np.pi # d=lambda/2
# 目標角度的理想空間相位延遲
phase_x_target = np.sin(np.radians(theta_0)) * np.cos(np.radians(phi_0))
phase_y_target = np.sin(np.radians(theta_0)) * np.sin(np.radians(phi_0))

AF_target_val = 0j
applied_phase_shifts = np.zeros((Nx, Ny)) # 紀錄真實套用的量化後相位

for ix in range(Nx):
    for iy in range(Ny):
        # 為了把波束打到目標角度，理論上需要的完美相位補償
        ideal_shift = -(ix * kd * phase_x_target + iy * kd * phase_y_target)
        
        # 晶片硬體的殘酷：只能選擇最近的檔位 (量化)
        if ps_bits is not None:
            step = 2 * np.pi / (2**ps_bits)
            actual_shift = np.round(ideal_shift / step) * step
        else:
            actual_shift = ideal_shift
            
        applied_phase_shifts[ix, iy] = actual_shift
        
        # 計算在「正對著目標角度」時，真實疊加出來的訊號向量
        target_spatial_phase = ix * kd * phase_x_target + iy * kd * phase_y_target
        AF_target_val += np.exp(1j * (target_spatial_phase + actual_shift))

# 量化損耗計算 (Quantization Loss)
AF_target_norm = np.abs(AF_target_val) / N_total
quantization_loss_db = 20 * np.log10(AF_target_norm + 1e-12)

# 共用的陣列增益與損耗計算
scan_loss_db = -10 * np.log10(np.cos(np.radians(theta_0))) if theta_0 < 89 else 20.0
split_stages = np.log2(N_total) if N_total > 0 else 0
backend_passive_loss_db = (split_stages * splitter_il_db) + ps_il_db 

# [核心更新] 陣列空間處理總增益 (包含量化損耗)
array_spatial_gain_dbi = 10 * np.log10(N_total) + elem_gain_dbi - scan_loss_db + quantization_loss_db 

# ==========================================
# 4. 側邊欄：Tx / Rx 模式切換與專屬參數
# ==========================================
st.sidebar.header("3. 工作模式切換 (Operation Mode)")
mode = st.sidebar.radio("選擇鏈路方向：", ["Tx 模式 (發射鏈路)", "Rx 模式 (接收鏈路)"])

# ---------------------------------------------------------
# Tx 模式專屬邏輯
# ---------------------------------------------------------
if mode == "Tx 模式 (發射鏈路)":
    st.sidebar.subheader("📡 Tx 射頻與 PA 設定")
    feed_power_dbm = st.sidebar.number_input("系統饋入點功率 (dBm)", value=0.0, step=1.0)
    pa_gain_db = st.sidebar.number_input("單一 PA 增益 (dB)", value=15.0, step=1.0)
    rad_eff_db = st.sidebar.slider("天線輻射效率 (dB)", -5.0, 0.0, -2.0, step=0.5)
    
    st.sidebar.subheader("🎯 3GPP Tx 規範驗證")
    min_eirp_dbm = st.sidebar.number_input("3GPP Min Peak EIRP 下限 (dBm)", value=22.4, step=0.1)
    max_trp_dbm = st.sidebar.number_input("3GPP Max TRP 上限 (dBm)", value=23.0, step=0.1)

    pin_pa_dbm = feed_power_dbm - (10 * np.log10(N_total) + backend_passive_loss_db)
    pout_pa_dbm = pin_pa_dbm + pa_gain_db
    power_at_antenna_element_dbm = pout_pa_dbm - tr_switch_il_db
    total_conducted_tx_dbm = power_at_antenna_element_dbm + 10 * np.log10(N_total)
    
    trp_dbm = total_conducted_tx_dbm + rad_eff_db
    eirp_dbm = total_conducted_tx_dbm + array_spatial_gain_dbi

    st.subheader("📊 Tx 鏈路預算與 3GPP 規範驗證")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("單一 PA 輸出功率", f"{pout_pa_dbm:.1f} dBm")
    c2.metric("抵達單一天線之功率", f"{power_at_antenna_element_dbm:.1f} dBm", delta=f"被 T/R 開關吃掉 {-tr_switch_il_db} dB", delta_color="inverse")
    c3.metric("陣列空間總增益", f"{array_spatial_gain_dbi:.2f} dBi", delta=f"移相器量化損耗 {quantization_loss_db:.2f} dB", delta_color="normal", help="受限於 RFIC 移相器位元數，無法完美對齊相位造成的增益損失。")
    c4.metric("陣列真實導通功率", f"{total_conducted_tx_dbm:.1f} dBm")

    st.write("")
    st.markdown("**(3) 🎯 3GPP Class 3 規格驗證 (Regulatory Compliance)**")
    c5, c6 = st.columns(2)
    with c5:
        trp_delta = trp_dbm - max_trp_dbm
        st.metric("📦 預估 TRP (總輻射功率)", f"{trp_dbm:.1f} dBm", delta=f"{trp_delta:+.1f} dB (vs Max TRP 上限)", delta_color="inverse")
    with c6:
        eirp_delta = eirp_dbm - min_eirp_dbm
        st.metric("🚀 預估 Peak EIRP", f"{eirp_dbm:.1f} dBm", delta=f"{eirp_delta:+.1f} dB (vs Min EIRP 下限)", delta_color="normal")

# ---------------------------------------------------------
# Rx 模式專屬邏輯
# ---------------------------------------------------------
elif mode == "Rx 模式 (接收鏈路)":
    st.sidebar.subheader("🎧 Rx 射頻與 LNA 設定")
    lna_gain_db = st.sidebar.number_input("單一 LNA 增益 (dB)", value=20.0, step=1.0)
    lna_nf_db = st.sidebar.number_input("單一 LNA 雜訊指數 (NF, dB)", value=3.0, step=0.1)
    bw_mhz = st.sidebar.number_input("系統頻寬 (MHz)", value=100.0, step=10.0)
    
    st.sidebar.subheader("🧩 系統解調門檻")
    snr_ideal_db = st.sidebar.number_input("理論 Baseband SNR (dB)", value=-1.0, step=0.5)
    margin_db = st.sidebar.number_input("硬體實現損耗 Margin (dB)", value=2.5, step=0.5)
    snr_min_db = snr_ideal_db + margin_db
    
    st.sidebar.subheader("🎯 3GPP Rx 規範驗證")
    target_eis_dbm = st.sidebar.number_input("目標 Peak EIS 上限 (dBm)", value=-88.0, step=0.5)

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
    c1.metric("1. 系統熱雜訊底噪", f"{kTB_dbm:.1f} dBm")
    c2.metric("2. 前端 T/R 開關損耗", f"{tr_switch_il_db:.1f} dB")
    c3.metric("3. 整機串聯 NF", f"{nf_cascaded_db:.2f} dB", delta=f"T/R與後端拖累 {nf_cascaded_db - lna_nf_db:.2f} dB", delta_color="inverse")
    c4.metric("4. 傳導靈敏度", f"{conducted_sens_dbm:.2f} dBm")

    st.write("")
    st.markdown("**(2) 空間輻射層面 (OTA & 3GPP 驗證)**")
    c5, c6 = st.columns(2)
    with c5:
        st.metric("📡 Rx 陣列空間總增益", f"+{array_spatial_gain_dbi:.2f} dBi", delta=f"移相器量化損耗 {quantization_loss_db:.2f} dB", delta_color="normal")
    with c6:
        eis_delta = peak_eis_dbm - target_eis_dbm
        st.metric("🎯 預估 Peak EIS", f"{peak_eis_dbm:.2f} dBm", delta=f"{eis_delta:+.2f} dB (vs 目標 EIS)", delta_color="inverse")

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
        # 使用剛剛計算好的「真實套用之量化相位 (applied_phase_shifts)」來算 3D 空間響應
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

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z, surfacecolor=Gain_Matrix_dBi, 
    colorscale='Jet', cmin=cmin_val, cmax=array_spatial_gain_dbi,
    colorbar=dict(title="Gain (dBi)", thickness=20)
)])

fig.update_layout(
    title=f"3D Radiation Pattern (Target: θ={theta_0}°, φ={phi_0}°) - TDD 真實晶片場型",
    scene=dict(
        xaxis=dict(title='X', range=[-1, 1], showbackground=False),
        yaxis=dict(title='Y', range=[-1, 1], showbackground=False),
        zaxis=dict(title='Z (正前方)', range=[0, 1], showbackground=False),
        aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    width=800, height=600, margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig, use_container_width=True)