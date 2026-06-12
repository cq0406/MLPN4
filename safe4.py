import hashlib
import json
import os

class GlobalPatientDB:
    def __init__(self, filepath="global_patient_db.json"):
        self.filepath = filepath
        self.data = self.load()

    def load(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_record(self, record):
        self.data.append(record)
        self.save()

    def get_all(self):
        return self.data

global_db = GlobalPatientDB()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, root_scalar
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import uuid
import joblib
from datetime import date as dt_date

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os

# 重新扫描字体
fm._load_fontmanager(try_read_cache=False)

# 使用上传的微软雅黑字体（绝对路径 /mount/src/mlpn/）
font_path = '/mount/src/mlpn/msyh.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

else:
    # 如果未找到，回退到系统字体（但不应发生）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="美罗培南智能给药", layout="wide")
# ==================== 多用户权限与全局数据库 ====================
import hashlib
import json

class GlobalPatientDB:
    def __init__(self, filepath="global_patient_db.json"):
        self.filepath = filepath
        self.data = self.load()

    def load(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_record(self, record):
        self.data.append(record)
        self.save()

    def get_all(self):
        return self.data

global_db = GlobalPatientDB()

# ---------- 身份选择 ----------
if "user_role" not in st.session_state:
    st.session_state.user_role = None

if st.session_state.user_role is None:
    st.title("💊 美罗培南智能化精准给药系统")
    role = st.radio("请选择身份", ["普通用户", "管理员"], horizontal=True)
    if role == "管理员":
        pwd = st.text_input("请输入管理员密码", type="password")
        if st.button("登录"):
            # MD5 hash of "admin123"
            if hashlib.md5(pwd.encode()).hexdigest() == "0192023a7bbd73250516f069df18b500":
                st.session_state.user_role = "admin"
                st.session_state.patient_db = global_db.get_all()
                st.rerun()
            else:
                st.error("密码错误")
    else:
        if st.button("进入系统"):
            st.session_state.user_role = "user"
            if "patient_db" not in st.session_state:
                st.session_state.patient_db = []
            st.rerun()
    st.stop()

# ---------- 管理员退出按钮 ----------
if st.session_state.user_role == "admin":
    with st.sidebar:
        st.markdown("### 🔒 管理员模式")
        if st.button("退出管理员"):
            st.session_state.user_role = None
            st.session_state.patient_db = []
            st.rerun()
import os

# 自动加载已训练好的模型（如果文件存在）
def load_models():
    if os.path.exists("pk_model.pkl"):
        st.session_state.pk_model = joblib.load("pk_model.pkl")
    if os.path.exists("outcome_model.pkl"):
        st.session_state.outcome_model = joblib.load("outcome_model.pkl")
    if os.path.exists("pk_features.pkl"):
        st.session_state.pk_features = joblib.load("pk_features.pkl")
    if os.path.exists("outcome_encoder.pkl"):
        st.session_state.outcome_encoder = joblib.load("outcome_encoder.pkl")

load_models()
# ---------- 全局状态初始化 ----------
if "patient_db" not in st.session_state:
    st.session_state.patient_db = []

for rec in st.session_state.patient_db:
    rec.setdefault("record_id", str(uuid.uuid4()))
    rec.setdefault("编号", rec.get("病号", ""))
    if "日期" in rec:
        rec["日期"] = str(rec["日期"])[:10].strip()
    else:
        rec["日期"] = ""
    rec.setdefault("滴注速度", None)
    rec.setdefault("疗效", "")

if "show_stats" not in st.session_state: st.session_state.show_stats = False
if "show_bayes" not in st.session_state: st.session_state.show_bayes = False
if "show_validation" not in st.session_state: st.session_state.show_validation = False
if "show_efficacy" not in st.session_state: st.session_state.show_efficacy = False
if "show_montecarlo" not in st.session_state: st.session_state.show_montecarlo = False
if "show_lasso" not in st.session_state: st.session_state.show_lasso = False
if "show_training" not in st.session_state: st.session_state.show_training = False

if "pathogens" not in st.session_state: st.session_state.pathogens = [""]
if "mics" not in st.session_state: st.session_state.mics = [1]

# ---------- 工具函数（全部保留，仅展示部分关键函数） ----------
def simulate_conc_full(dose, interval, cl=12.0, vd=0.25, weight=65, infusion_time=0.5):
    """
    一室模型多次静脉输注叠加模拟
    dose: 单次剂量 (g)
    interval: 给药间隔 (h)
    cl: 清除率 (L/h)
    vd: 表观分布容积 (L/kg)
    weight: 体重 (kg)
    infusion_time: 每次输注时长 (h)
    返回: 时间数组 (h), 浓度数组 (mg/L)
    """
    vd_total = vd * weight          # 总体分布容积 (L)
    ke = cl / vd_total              # 消除速率常数 (1/h)
    dose_mg = dose * 1000           # 剂量转换为 mg

    # 模拟总时长：至少 10 个给药间隔，确保稳态
    total_hours = max(48, interval * 10)
    dt = 0.05                       # 时间步长，生成平滑曲线
    t = np.arange(0, total_hours + dt, dt)
    conc = np.zeros_like(t)

    n_doses = int(np.ceil(total_hours / interval)) + 1

    for i in range(n_doses):
        t_start = i * interval
        t_end_inf = t_start + infusion_time
        if t_start > total_hours:
            break

        # 输注期内的索引
        mask_inf = (t >= t_start) & (t < t_end_inf)
        # 输注结束后的索引
        mask_post = t >= t_end_inf

        # 输注速率 (mg/h)
        inf_rate = dose_mg / infusion_time

        # 输注期浓度贡献
        if np.any(mask_inf):
            conc[mask_inf] += (inf_rate / (ke * vd_total)) * (1 - np.exp(-ke * (t[mask_inf] - t_start)))

        # 输注后衰减
        if np.any(mask_post):
            c_end_inf = (inf_rate / (ke * vd_total)) * (1 - np.exp(-ke * infusion_time))
            conc[mask_post] += c_end_inf * np.exp(-ke * (t[mask_post] - t_end_inf))

    return t, conc

def is_severe(体温, WBC):
    try:
        t = float(体温) if 体温 not in ["无", ""] else 0
        w = float(WBC) if WBC not in ["无", ""] else 0
        return t > 39 or w > 18
    except:
        return False

def is_resistant(mic_list):
    try:
        return any(m > 8 for m in mic_list)
    except:
        return False

def predict_dose(weight, inf_sites, severe, resistant):
    try:
        w = float(weight) if weight not in ["无", ""] else 65
    except:
        w = 65
    base = 1.0
    # 检查是否有中枢神经系统感染
    has_cns = False
    if isinstance(inf_sites, list):
        has_cns = "中枢感染" in inf_sites
    else:
        has_cns = (inf_sites == "中枢感染")
    if severe or resistant or has_cns:
        base = 1.5
    if w > 80:
        base = 2.0
    elif w < 50:
        base = 0.5
    return base

def get_interval(inf_sites, severe, resistant):
    has_cns = False
    if isinstance(inf_sites, list):
        has_cns = "中枢神经系统感染" in inf_sites
    else:
        has_cns = (inf_sites == "中枢神经系统感染")
    return 8 if severe or resistant or has_cns else 12

def recommend_infusion_time(inf_sites, severe, resistant, mic_list):
    try:
        max_mic = max(mic_list) if mic_list else 1
    except:
        max_mic = 1
    has_cns = False
    if isinstance(inf_sites, list):
        has_cns = "中枢神经系统感染" in inf_sites
    else:
        has_cns = (inf_sites == "中枢神经系统感染")
    if (severe or resistant or has_cns) and max_mic >= 4:
        return 3.0
    else:
        return 0.5

def simulate_conc(dose, interval, cl=12.0, vd=0.25, weight=65, infusion_time=0.5):
    """
    标准一室模型多次静脉输注，模拟至充分消除。
    dose: g
    interval: h
    cl: L/h
    vd: L/kg (总容积 = vd * weight)
    weight: kg
    infusion_time: h
    """
    Vd_total = vd * weight          # L
    ke = cl / Vd_total              # 1/h
    if ke <= 0:
        ke = 0.01
    dose_mg = dose * 1000           # mg

    # 模拟时长：至少48h，且至少10个Half-life，确保消除相完整
    t_half = np.log(2) / ke
    total_hours = max(48, 10 * t_half, interval * 10)
    dt = 0.05
    t = np.arange(0, total_hours + dt, dt)
    conc = np.zeros_like(t)

    n_doses = int(np.ceil(total_hours / interval)) + 1

    for i in range(n_doses):
        t_start = i * interval
        t_end_inf = t_start + infusion_time
        if t_start > total_hours:
            break

        # 输注期掩码
        mask_inf = (t >= t_start) & (t < t_end_inf)
        # 输注后掩码
        mask_post = t >= t_end_inf

        if np.any(mask_inf):
            # 输注速率 (mg/h)
            inf_rate = dose_mg / infusion_time
            # 标准公式：C(t) = (inf_rate / (ke * Vd)) * (1 - exp(-ke*(t-t_start)))
            conc[mask_inf] += (inf_rate / (ke * Vd_total)) * (1 - np.exp(-ke * (t[mask_inf] - t_start)))

        if np.any(mask_post):
            # 输注结束时刻的浓度
            c_end_inf = (inf_rate / (ke * Vd_total)) * (1 - np.exp(-ke * infusion_time))
            # 消除段：C(t) = C_end_inf * exp(-ke*(t - t_end_inf))
            conc[mask_post] += c_end_inf * np.exp(-ke * (t[mask_post] - t_end_inf))

    return t, conc

def calc_infusion_rate(dose_g, infusion_time_h):
    """计算滴注速度 (mL/min)，假设 1g 药物稀释至 100 mL"""
    volume_ml = dose_g * 100  # mL
    infusion_time_min = infusion_time_h * 60  # 分钟
    if infusion_time_min > 0:
        return round(volume_ml / infusion_time_min, 2)
    else:
        return 0.0

def calculate_ft_mic(conc, mic_values):
    try:
        return [round((np.sum(conc >= mic)/len(conc))*100, 1) for mic in mic_values]
    except:
        return [0.0]*len(mic_values)

def calculate_ft_mic(conc, mic_vals):
    return [round(np.sum(conc>=m)/len(conc)*100,1) for m in mic_vals]

def bayesian_estimate(obs_t, obs_c, dose, interval, weight, prior_cl=12.0, prior_vd=0.25):
    def pk_model(params, t_obs, dose, interval, weight):
        cl, vd = params
        t_sim, conc_sim = simulate_conc(dose, interval, cl, vd, weight)
        return np.interp(float(t_obs), t_sim, conc_sim)
    def obj(params):
        cl, vd = params
        pred = pk_model(params, obs_t, dose, interval, weight)
        sigma = max(0.2*pred, 0.1)
        lik = ((obs_c - pred)/sigma)**2
        prior = ((cl-prior_cl)/5)**2 + ((vd-prior_vd)/0.1)**2
        return lik + prior
    res = minimize(obj, x0=[prior_cl, prior_vd], bounds=[(1,30),(0.1,0.8)], method='L-BFGS-B')
    return (res.x[0], res.x[1]) if res.success else (prior_cl, prior_vd)

def monte_carlo_pta(dose, interval, weight, mic_target, n_sim=500,
                    cl_mean=12.0, cl_cv=0.3, vd_mean=0.25, vd_cv=0.15):
    np.random.seed(42)
    cl_sim = np.random.lognormal(np.log(cl_mean), cl_cv, n_sim)
    vd_sim = np.random.lognormal(np.log(vd_mean), vd_cv, n_sim)
    ft_target = 40
    success = 0
    for i in range(n_sim):
        t, conc = simulate_conc(dose, interval, cl_sim[i], vd_sim[i], weight)  # 删除 hours=48
        ft = (np.sum(conc >= mic_target) / len(conc)) * 100
        if ft >= ft_target:
            success += 1
    return (success / n_sim) * 100

def ckd_epi_2021(scr, age, gender):
    scr_mg = scr/88.4
    kappa = 0.7 if gender=="女" else 0.9
    alpha = -0.241 if gender=="女" else -0.302
    gf = 1.012 if gender=="女" else 1.0
    ratio = scr_mg/kappa
    return 142 * (min(ratio,1)**alpha) * (max(ratio,1)**-1.200) * (0.9938**age) * gf

def calc_crcl(scr, age, weight, gender):
    """
    计算肌酐清除率 (CrCl, mL/min) 使用 Cockcroft-Gault 公式
    scr: 血清肌酐 (μmol/L)
    age: 岁
    weight: kg
    gender: "男" 或 "女"
    """
    if scr <= 0 or age <= 0 or weight <= 0:
        return None
    # 转换为 mg/dL
    scr_mg = scr / 88.4
    crcl = ((140 - age) * weight) / (72 * scr_mg)
    if gender == "女":
        crcl *= 0.85
    return round(crcl, 1)

def calc_mdrd(scr, age, gender):
    """
    计算 eGFR (mL/min/1.73m²) 使用简化 MDRD 公式（中国人群修正）
    scr: 血清肌酐 (μmol/L)
    age: 岁
    gender: "男" 或 "女"
    """
    if scr <= 0 or age <= 0:
        return None
    # 转换为 mg/dL
    scr_mg = scr / 88.4
    egfr = 175 * (scr_mg ** -1.154) * (age ** -0.203)
    if gender == "女":
        egfr *= 0.742
    # 中国人群修正系数 1.233（可根据需要调整）
    egfr *= 1.233
    return round(egfr, 1)


def reverse_egfr_to_scr(egfr, age, gender):
    def f(scr): return ckd_epi_2021(scr, age, gender) - egfr
    try:
        sol = root_scalar(f, bracket=[10,1500], method='brentq')
        return round(sol.root,1) if sol.converged else None
    except: return None

# ========== 全局特征工程函数（供主页面和模型训练调用） ==========
def add_engineered_features(df):
    df = df.copy()
    # 统一“病原菌列表”和“感染部位”为列表
    if "病原菌列表" in df.columns:
        def safe_list(x):
            if isinstance(x, list): return x
            elif isinstance(x, str) and x.strip():
                return [v.strip() for v in x.replace('，', ',').split(',') if v.strip()]
            else: return []
        df["病原菌列表"] = df["病原菌列表"].apply(safe_list)
    if "感染部位" in df.columns:
        df["感染部位"] = df["感染部位"].apply(
            lambda x: x if isinstance(x, list) else ([v.strip() for v in str(x).replace('，', ',').split(',') if v.strip()] if pd.notna(x) and str(x).strip() else [])
        )
    # 体重默认值分性别
    if "体重" in df.columns:
        df["体重"] = pd.to_numeric(df["体重"], errors='coerce')
        male_mask = (df["性别"] == "男") & df["体重"].isna()
        female_mask = (df["性别"] == "女") & df["体重"].isna()
        unknown_mask = df["体重"].isna() & ~male_mask & ~female_mask
        df.loc[male_mask, "体重"] = 70.0
        df.loc[female_mask, "体重"] = 60.0
        df.loc[unknown_mask, "体重"] = 65.0
    else:
        df["体重"] = 65.0
    if "年龄" in df.columns:
        df["年龄"] = pd.to_numeric(df["年龄"], errors='coerce').fillna(45)
    else:
        df["年龄"] = 45
    if "身高" in df.columns:
        df["身高"] = pd.to_numeric(df["身高"], errors='coerce')
        male_mask = (df["性别"] == "男") & df["身高"].isna()
        female_mask = (df["性别"] == "女") & df["身高"].isna()
        df.loc[male_mask, "身高"] = 170; df.loc[female_mask, "身高"] = 160
        df["身高"] = df["身高"].fillna(165)
    else:
        df["身高"] = 165
    if "肌酐" in df.columns:
        df["肌酐"] = pd.to_numeric(df["肌酐"], errors='coerce').fillna(88)
    else:
        df["肌酐"] = 88

    height_m = df["身高"] / 100
    df["BMI"] = df["体重"] / (height_m ** 2)

    def calc_crcl(row):
        scr = row["肌酐"]; age = row["年龄"]; weight = row["体重"]
        gender = row.get("性别", "男")
        if scr <= 0 or weight <= 0 or age <= 0: return 0
        scr_mg = scr / 88.4
        crcl = ((140 - age) * weight) / (72 * scr_mg)
        if gender == "女": crcl *= 0.85
        return round(crcl, 1)
    df["CrCl"] = df.apply(calc_crcl, axis=1)

    if "剂量" in df.columns and "间隔" in df.columns:
        dose = pd.to_numeric(df["剂量"], errors='coerce')
        interval = pd.to_numeric(df["间隔"], errors='coerce')
        df["日总剂量"] = np.where(interval > 0, dose * (24 / interval), np.nan)
    else:
        df["日总剂量"] = np.nan
    if "剂量" in df.columns:
        dose = pd.to_numeric(df["剂量"], errors='coerce')
        df["剂量_体重"] = np.where(df["体重"] > 0, dose * 1000 / df["体重"], np.nan)
    else:
        df["剂量_体重"] = np.nan
    df["是否老年"] = (df["年龄"] >= 65).astype(int)

    # 基于体重的预测清除率（安全处理缺失列）
    if "eGFR" in df.columns:
        egfr_temp = pd.to_numeric(df["eGFR"], errors='coerce').fillna(90)
    else:
        egfr_temp = 90  # 默认 eGFR

    if "体重" in df.columns:
        weight_for_cl = pd.to_numeric(df["体重"], errors='coerce').fillna(60)
    else:
        weight_for_cl = 60

    weight_based_cl_pred = 7.0 + 0.35 * ((weight_for_cl - 60.0) / 5.0)
    weight_based_cl_pred = weight_based_cl_pred.clip(lower=4.0)
    df["CL_pred"] = weight_based_cl_pred * (egfr_temp / 90.0) ** 0.75

    df["CrCl_x_老年"] = df["CrCl"] * df["是否老年"]

    if "感染部位" in df.columns:
        df["感染部位_首位"] = df["感染部位"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
    else:
        df["感染部位_首位"] = None

    if "MIC列表" in df.columns:
        df["是否耐药"] = df["MIC列表"].apply(
            lambda x: 1 if isinstance(x, list) and any(m > 8 for m in x) else 0
        )
    else:
        df["是否耐药"] = 0

    return df
# ----------------------------- 标题与导航 -----------------------------
st.title("💊 美罗培南智能化精准给药系统")

# ---------- 优化后的导航栏 ----------
nav_cols = st.columns(8, gap="small")
nav_buttons = [
    ("📊 统计分析", "show_stats"),
    ("🧬 贝叶斯估算", "show_bayes"),
    ("✅ 结果验证", "show_validation"),
    ("📈 疗效评估", "show_efficacy"),
    ("🎲 蒙特卡洛", "show_montecarlo"),
    ("📈 LASSO回归", "show_lasso"),
    ("🤖 模型训练", "show_training"),
    ("📋 返回录入", "return"),
]

# 需要关闭的其他模块键名
other_modules = ["show_stats","show_bayes","show_validation","show_efficacy","show_montecarlo","show_lasso","show_training"]

for idx, (label, key) in enumerate(nav_buttons):
    with nav_cols[idx]:
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            if key == "return":
                # 返回录入：关闭所有模块，并重置药物选择
                for mod in other_modules:
                    st.session_state[mod] = False
                st.session_state.selected_drug = None
            else:
                st.session_state[key] = True
                for other in other_modules:
                    if other != key:
                        st.session_state[other] = False
            st.rerun()

st.divider()
# ======================== 蒙特卡洛模拟模块 ========================
if st.session_state.show_montecarlo:
    st.subheader("🎲 蒙特卡洛模拟 - 达标概率(PTA)分析")
    if st.button("← 返回录入页面"):
        st.session_state.show_montecarlo = False
        st.rerun()
    col1, col2, col3 = st.columns(3)
    with col1:
        dose_mc = st.number_input("剂量 (g)", min_value=0.5, value=1.0, step=0.25)
        interval_mc = st.selectbox("间隔 (h)", [6, 8, 12], index=1)
    with col2:
        weight_mc = st.number_input("体重 (kg)", min_value=30, value=70, step=1)
        mic_target = st.selectbox("目标MIC (mg/L)", [0.5, 1, 2, 4, 8, 16], index=2)
    with col3:
        n_sim = st.number_input("模拟次数", min_value=100, max_value=2000, value=500, step=100)
        cl_cv = st.slider("CL变异系数 (%)", 10, 50, 30) / 100
        vd_cv = st.slider("Vd变异系数 (%)", 5, 30, 15) / 100

    if st.button("运行蒙特卡洛模拟"):
        with st.spinner("模拟中，请稍候..."):
            pta = monte_carlo_pta(dose_mc, interval_mc, weight_mc, mic_target,
                                  n_sim=n_sim, cl_mean=12.0, cl_cv=cl_cv,
                                  vd_mean=0.25, vd_cv=vd_cv)
        st.metric("达标概率 (PTA)", f"{pta:.1f}%")
        if pta >= 90:
            st.success("该方案达标概率优秀（≥90%）")
        elif pta >= 80:
            st.info("该方案达标概率良好（80-90%）")
        else:
            st.warning("该方案达标概率不足，建议调整剂量或间隔")

        st.markdown("### 模拟样本曲线（前20条）")
        np.random.seed(42)
        fig, ax = plt.subplots(figsize=(8,4))
        for i in range(min(20, n_sim)):
            cl_s = np.random.lognormal(np.log(12.0), cl_cv)
            vd_s = np.random.lognormal(np.log(0.25), vd_cv)
            t, conc = simulate_conc(dose_mc, interval_mc, cl_s, vd_s, weight_mc)
            ax.plot(t, conc, alpha=0.15, color='blue')
        ax.axhline(mic_target, ls='--', color='red', label=f'MIC={mic_target}')
        ax.set_xlabel(' (h)')
        ax.set_ylabel('浓度 (mg/L)')
        ax.set_title('蒙特卡洛模拟浓度曲线')
        ax.legend()
        st.pyplot(fig)
    st.stop()

# ======================== LASSO回归模块 ========================
if st.session_state.show_lasso:
    st.subheader("📈 LASSO回归建模 - 剂量预测/因素筛选")
    if st.button("← 返回录入页面"):
        st.session_state.show_lasso = False
        st.rerun()

    df = pd.DataFrame(st.session_state.patient_db)
    if len(df) < 10:
        st.warning("病Count不足（至少需要10例），请先录入更多数据或导入Excel")
    else:
        st.markdown(f"当前数据库病Count：{len(df)}")

        # ---------- 特征列定义（主页面数值字段） ----------
        feature_cols = [
            "年龄", "身高", "体重", "体温", "心率", "肌酐", "eGFR", "尿酸", "尿素",
            "总胆红素", "直接胆红素", "间接胆红素", "总蛋白", "白蛋白", "球蛋白",
            "丙氨酸氨基转移酶ALT", "AST", "碱性磷酸酶", "谷氨酰氨基转移酶",
            "白细胞(WBC)", "中性粒细胞", "血小板计数", "CRP", "血清淀粉样C蛋白SAA", "PCT",
            "滴注速度"   # 如果有则纳入，没有则跳过
        ]
        # 只保留实际存在的特征列
        available_features = [c for c in feature_cols if c in df.columns]

        if not available_features:
            st.error("数据库中没有可用的数值特征，请先录入数据。")
        else:
            # ---------- 数据预处理：智能填充缺失值，最大限度保留样本 ----------
            # 复制DataFrame，仅处理需要的列和目标
            data_df = df[available_features].copy()
            target_series = pd.to_numeric(df["剂量"], errors='coerce')

            # 1. 处理目标变量：剔除剂量缺失的记录
            valid_mask = target_series.notna()
            data_df = data_df[valid_mask]
            target_series = target_series[valid_mask]

            # 2. 对每个特征进行缺失值填充（中位数填充，若中位数不可得则用0）
            for col in available_features:
                col_data = pd.to_numeric(data_df[col], errors='coerce')
                if col_data.notna().sum() > 0:
                    median_val = col_data.median()
                    data_df[col] = col_data.fillna(median_val)
                else:
                    data_df[col] = col_data.fillna(0)

            # 3. 最终确保所有值为数值，无NaN
            X = data_df.astype(float).values
            y = target_series.astype(float).values

            # 再次过滤掉目标为NaN的行（已在第一步保证，但以防万一）
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]

            if len(X) == 0:
                st.error("无有效数值数据，请检查录入信息")
            else:
                st.success(f"可用样本量：{len(X)}（已自动填充缺失特征，仅剔除剂量缺失记录）")

                alpha = st.slider("正则化强度 α", 0.001, 1.0, 0.1, 0.01)

                if st.button("训练LASSO模型"):
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )

                    lasso = Lasso(alpha=alpha, max_iter=10000)
                    lasso.fit(X_train, y_train)

                    score_train = lasso.score(X_train, y_train)
                    score_test = lasso.score(X_test, y_test)
                    st.write(f"训练集 R² = {score_train:.3f}，测试集 R² = {score_test:.3f}")

                    # 系数展示
                    coef_df = pd.DataFrame({
                        "特征": available_features,
                        "系数": lasso.coef_
                    }).sort_values("系数", key=abs, ascending=False)
                    st.dataframe(coef_df)

                    # 非零系数图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    nonzero = coef_df[coef_df["系数"] != 0]
                    if not nonzero.empty:
                        ax.barh(nonzero["特征"], nonzero["系数"])
                        ax.set_xlabel("系数值")
                        ax.set_title("LASSO非零系数")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        st.pyplot(fig)
                    else:
                        st.info("当前 α 下所有系数被压缩为零，请减小 α 值。")

                    # 新患者剂量预测
                    st.markdown("### 对新患者进行剂量预测")
                    col_input = st.columns(3)
                    input_vals = []
                    for i, feat in enumerate(available_features):
                        with col_input[i % 3]:
                            val = st.number_input(feat, value=0.0, step=0.1, format="%.2f")
                            input_vals.append(val)

                    if st.button("预测剂量"):
                        input_arr = np.array(input_vals).reshape(1, -1)
                        input_scaled = scaler.transform(input_arr)
                        pred_dose = lasso.predict(input_scaled)[0]
                        st.metric("预测剂量 (g)", f"{max(0.25, pred_dose):.2f}")
    st.stop()

# ======================== 贝叶斯估算模块 ========================
if st.session_state.show_bayes:
    st.subheader("🧬 贝叶斯个体化参数估算")
    if st.button("← 返回录入页面"):
        st.session_state.show_bayes = False
        st.rerun()
    st.markdown("基于治疗药物监测(TDM)浓度反馈，估算个体清除率CL与分布容积Vd")
    col1, col2 = st.columns(2)
    with col1:
        obs_time = st.number_input("采血时间（给药后小时）", min_value=0.0, value=2.0, step=0.5)
        obs_conc = st.number_input("实测浓度（mg/L）", min_value=0.0, value=15.0, step=0.1)
    with col2:
        dose_bayes = st.number_input("给药剂量（g）", min_value=0.25, value=1.0, step=0.25)
        interval_bayes = st.number_input("给药间隔（h）", min_value=6, value=8, step=2)
        weight_bayes = st.number_input("体重（kg）", min_value=30, value=70, step=1)

    prior_cl = st.slider("先验清除率 CL (L/h)", 5.0, 25.0, 12.0, 0.5)
    prior_vd = st.slider("先验分布容积 Vd (L/kg)", 0.1, 0.5, 0.25, 0.01)

    if st.button("运行贝叶斯估算"):
        cl_est, vd_est = bayesian_estimate(obs_time, obs_conc, dose_bayes, interval_bayes, weight_bayes,
                                           prior_cl, prior_vd)
        st.success(f"估算结果：CL = {cl_est:.2f} L/h，Vd = {vd_est:.3f} L/kg")

        # 使用修复后的 simulate_conc（叠加达稳态）
        t_sim, conc_sim = simulate_conc(dose_bayes, interval_bayes, cl_est, vd_est, weight_bayes)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(t_sim, conc_sim, 'b-', label='个体预测')
        ax.scatter([obs_time], [obs_conc], c='r', s=80, label='实测浓度')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('浓度 (mg/L)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### 剂量调整建议")
        pred_at_time = np.interp(obs_time, t_sim, conc_sim)
        ratio = obs_conc / pred_at_time if pred_at_time > 0 else 1.0
        if ratio < 0.7:
            new_dose = dose_bayes * (1.3 if ratio < 0.5 else 1.15)
            st.warning(f"实测浓度低于预测，建议增加剂量至 {new_dose:.2f} g")
        elif ratio > 1.5:
            new_dose = dose_bayes * 0.7
            st.warning(f"实测浓度过高，建议减少剂量至 {new_dose:.2f} g")
        else:
            st.success("当前剂量适宜，无需调整")
    st.stop()
# ----------------------------- 结果验证模块（完整版） -----------------------------
if st.session_state.show_validation:
    st.subheader("✅ 模型结果验证")
    if st.button("← 返回录入页面"):
        st.session_state.show_validation = False
        st.rerun()
    st.markdown("输入患者编号及TDM实测浓度，对比系统Predicted Conc.与实际浓度，评估模型准确性。")

    验证编号 = st.text_input("患者编号", placeholder="例如：P001")
    col1, col2 = st.columns(2)
    with col1:
        actual_c3h = st.text_input("实测第5剂前3h浓度 (mg/L)", placeholder="无/数值")
    with col2:
        actual_c05h = st.text_input("实测第5剂前0.5h浓度 (mg/L)", placeholder="无/数值")

    if st.button("开始验证"):
        if not 验证编号:
            st.warning("请输入患者编号")
        else:
            df_pat = pd.DataFrame(st.session_state.patient_db)
            if df_pat.empty:
                st.error("数据库无记录")
            else:
                df_pat = df_pat[df_pat["编号"] == 验证编号]
                if df_pat.empty:
                    st.error(f"未找到编号为 {验证编号} 的患者记录")
                else:
                    record = df_pat.iloc[-1].to_dict()
                    st.success(f"找到患者 {验证编号}，年龄：{record.get('年龄','未知')}，感染部位：{record.get('感染部位','未知')}")

                    dose_val = record.get("剂量")
                    interval_val = record.get("间隔")
                    inf_time = record.get("Infusion Time", 0.5)
                    weight_val = record.get("体重", 65)
                    try:
                        wt = float(weight_val) if weight_val not in ["无", ""] else 65
                    except:
                        wt = 65

                    if dose_val is None or interval_val is None:
                        st.error("该记录缺少剂量或间隔信息，无法验证")
                    else:
                        cl_typical, vd_typical = 12.0, 0.25
                        # 完整模拟
                        t_full, conc_full = simulate_conc(
                            float(dose_val), float(interval_val),
                            cl=cl_typical, vd=vd_typical, weight=wt,
                            infusion_time=float(inf_time)
                        )

                        interval_float = float(interval_val)
                        # 计算第5剂前3h和0.5h的时刻（相对于给药开始时间）
                        t_3h_abs = 4 * interval_float - 3.0
                        t_05h_abs = 4 * interval_float - 0.5
                        pred_3 = np.interp(t_3h_abs, t_full, conc_full)
                        pred_05 = np.interp(t_05h_abs, t_full, conc_full)

                        # 解析实测值
                        act_3 = None
                        act_05 = None
                        if actual_c3h and actual_c3h.strip() not in ["无", ""]:
                            try: act_3 = float(actual_c3h.strip())
                            except: pass
                        if actual_c05h and actual_c05h.strip() not in ["无", ""]:
                            try: act_05 = float(actual_c05h.strip())
                            except: pass

                        # 表格对比
                        st.markdown("### 浓度对比")
                        data_comp = {
                            "时间点": ["第5剂前3h", "第5剂前0.5h"],
                            "Predicted Conc. (mg/L)": [f"{pred_3:.2f}", f"{pred_05:.2f}"],
                            "实测浓度 (mg/L)": [f"{act_3:.2f}" if act_3 else "未填",
                                               f"{act_05:.2f}" if act_05 else "未填"]
                        }
                        st.table(pd.DataFrame(data_comp))

                        # ---------- 曲线绘制：截取第5次给药间隔（稳态周期） ----------
                        start_cycle = 4 * interval_float
                        end_cycle = 5 * interval_float
                        if end_cycle > t_full[-1]:
                            end_cycle = t_full[-1]
                            start_cycle = max(0, end_cycle - interval_float)
                        start_idx = np.searchsorted(t_full, start_cycle)
                        end_idx = np.searchsorted(t_full, end_cycle)
                        t_cycle = t_full[start_idx:end_idx] - start_cycle
                        conc_cycle = conc_full[start_idx:end_idx]

                        # 预测时间点（相对于周期开始）
                        t_pred_3h_rel = interval_float - 3.0
                        t_pred_05h_rel = interval_float - 0.5

                        st.markdown("### Predicted Conc.曲线与实测点（第5次给药周期）")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(t_cycle, conc_cycle, 'b-', lw=2, label='Predicted Conc.')
                        # 标注预测点
                        ax.scatter([t_pred_3h_rel, t_pred_05h_rel], [pred_3, pred_05],
                                   color='blue', s=60, zorder=5, label='预测点')
                        # 标注实测点（如果有）
                        if act_3 is not None:
                            ax.scatter([t_pred_3h_rel], [act_3], color='red', s=80, marker='s', zorder=5,
                                       label=f'Measured 3h: {act_3:.2f}')
                        if act_05 is not None:
                            ax.scatter([t_pred_05h_rel], [act_05], color='green', s=80, marker='^', zorder=5,
                                       label=f'Measured 0.5h: {act_05:.2f}')
                        # 垂直虚线标记时间点
                        ax.axvline(t_pred_3h_rel, ls=':', color='gray', alpha=0.7)
                        ax.axvline(t_pred_05h_rel, ls=':', color='gray', alpha=0.7)
                        ax.set_xlabel('Time (h)')
                        ax.set_ylabel('浓度 (mg/L)')
                        ax.set_title('美罗培南第5次给药周期稳态曲线')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # 偏差分析
                        st.markdown("### 偏差分析")
                        if act_05 is not None and pred_05 > 0:
                            bias_05 = (act_05 - pred_05) / pred_05 * 100
                            st.metric("0.5h浓度相对偏差", f"{bias_05:.1f}%")
                            if abs(bias_05) > 30:
                                st.warning("偏差较大，建议重新评估PK参数或检查TDM采样时间")
                            else:
                                st.success("模型预测良好（偏差 ≤30%）")
                        if act_3 is not None and pred_3 > 0:
                            bias_3 = (act_3 - pred_3) / pred_3 * 100
                            st.metric("3h浓度相对偏差", f"{bias_3:.1f}%")
                            if abs(bias_3) > 30:
                                st.warning("3h浓度偏差较大，请检查采样时间")
    st.stop()

# ======================== 疗效评估模块 ========================
if st.session_state.show_efficacy:
    st.subheader("📈 疗效评估（用药前后对比 + 综合评分）")
    if st.button("← 返回录入页面"):
        for key in ["eval_started", "eval_patient", "eval_date_before", "eval_date_after", "rec_before", "rec_after"]:
            st.session_state.pop(key, None)
        st.session_state.show_efficacy = False
        st.rerun()

    st.markdown("输入患者编号，选择用药前后日期，开始评估。")

    患者编号 = st.text_input("患者编号", placeholder="例如：P001").strip()

    if "eval_started" not in st.session_state:
        st.session_state.eval_started = False
    if "last_patient" not in st.session_state or st.session_state.last_patient != 患者编号:
        st.session_state.eval_started = False
        st.session_state.last_patient = 患者编号

    if not st.session_state.eval_started:
        df = pd.DataFrame(st.session_state.patient_db)
        if "日期" in df.columns:
            df["日期"] = df["日期"].astype(str).str.strip()
        else:
            st.error("数据库中缺少“日期”字段。")
            st.stop()

        df_pat = df[df["编号"] == 患者编号].copy()
        if df_pat.empty:
            st.warning(f"未找到编号为 {患者编号} 的患者记录")
        else:
            raw_dates = df_pat["日期"].dropna().unique()
            dates = sorted(set(raw_dates))
            if not dates:
                st.error("该患者没有有效日期记录。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    用药前日期 = st.selectbox("选择用药前日期", dates, index=0, key="date_before_select")
                with col2:
                    用药后日期 = st.selectbox("选择用药后日期", dates, index=min(1, len(dates)-1), key="date_after_select")

                if st.button("✅ 开始评估"):
                    before = df_pat[df_pat["日期"] == 用药前日期]
                    after = df_pat[df_pat["日期"] == 用药后日期]
                    if before.empty or after.empty:
                        st.error("未找到所选日期的记录。")
                    else:
                        st.session_state.rec_before = before.iloc[0].to_dict()
                        st.session_state.rec_after = after.iloc[0].to_dict()
                        st.session_state.eval_started = True
                        st.session_state.eval_patient = 患者编号
                        st.session_state.eval_date_before = 用药前日期
                        st.session_state.eval_date_after = 用药后日期
                        st.rerun()
    else:
        rec_before = st.session_state.rec_before
        rec_after = st.session_state.rec_after
        患者编号 = st.session_state.eval_patient
        用药后日期 = st.session_state.eval_date_after

        st.success(f"当前评估：{患者编号}，用药前 {st.session_state.eval_date_before} → 用药后 {用药后日期}")

        def get_val(col):
            try:
                v = rec_before.get(col) if col in rec_before else rec_after.get(col)
                if v is None or v == "":
                    return None
                return float(v)
            except:
                return None

        if "efficacy_scores" not in st.session_state:
            st.session_state.efficacy_scores = {}

        # ==================== 一、临床症状与体征（30分） ====================
        st.subheader("一、临床症状与体征（满分 30 分）")
        with st.expander("评分", expanded=True):
            st.caption("全身症状（10分）：0分（无改善/加重）| 3分（症状减轻，需对症）| 6分（症状明显减轻，无需对症）| 10分（发热、寒战、乏力完全消失）")
            s1 = st.select_slider("全身症状", options=[0, 3, 6, 10], value=st.session_state.efficacy_scores.get("全身症状", 0), key="s1")

            st.caption("局部症状（10分）：选择受累部位，每个部位单独评分，总分合计10分")
            # 感染部位选项（与主页面一致）
            site_options = [
                "未查明", "泌尿系统感染", "手术部位感染", "血流感染", "上呼吸道感染", "下呼吸道感染",
                "骨关节感染", "中枢神经系统感染", "心血管系统感染", "消化系统感染", "生殖系统感染",
                "皮肤软组织感染", "全身性感染", "眼/耳/鼻/喉/口腔感染"
            ]
            # 多选局部症状部位
            local_sites = st.multiselect("选择受累部位", site_options, key="local_sites")
            # 动态生成评分条
            if local_sites:
                num_sites = len(local_sites)
                per_site_max = 10 // num_sites  # 每个部位满分
                remainder = 10 % num_sites      # 余数分配给第一个部位
                local_total = 0
                for i, site in enumerate(local_sites):
                    max_score = per_site_max + (1 if i == 0 else 0) * remainder  # 第一个部位多给余数分
                    options = list(range(0, max_score + 1))
                    score = st.select_slider(
                        f"{site}",
                        options=options,
                        value=min(st.session_state.efficacy_scores.get(f"local_{site}", 0), max_score),
                        key=f"local_{site}"
                    )
                    st.session_state.efficacy_scores[f"local_{site}"] = score
                    local_total += score
                # 显示局部总分
                st.caption(f"局部症状总分：{local_total}/10 分")
                s2 = local_total
            else:
                s2 = 0
                st.caption("局部症状总分：0/10 分（未选择部位）")

            st.caption("心率（3分）：0分（无改善）| 1分（上升或下降 <20%）| 2分（上升或下降 ≥20%）| 3分（恢复正常）")
            hr = st.select_slider("心率", options=[0, 1, 2, 3], value=st.session_state.efficacy_scores.get("心率", 0), key="hr")

            st.caption("血压（3分）：0分（无改善）| 1分（改善）| 2分（明显改善）| 3分（恢复正常）")
            bp = st.select_slider("血压", options=[0, 1, 2, 3], value=st.session_state.efficacy_scores.get("血压", 0), key="bp")

            st.caption("体温（4分）：0分（无改善）| 1分（上升或下降 <1℃）| 2分（上升或下降 ≥1℃）| 4分（恢复正常）")
            temp = st.select_slider("体温", options=[0, 1, 2, 4], value=st.session_state.efficacy_scores.get("体温", 0), key="temp")

            st.session_state.efficacy_scores["全身症状"] = s1
            # s2 已通过局部总分计算，不需要单独存储到 scores 中
            st.session_state.efficacy_scores["心率"] = hr
            st.session_state.efficacy_scores["血压"] = bp
            st.session_state.efficacy_scores["体温"] = temp

        # ==================== 二、实验室炎症指标（25分） ====================
        st.subheader("二、实验室炎症指标（满分 25 分）")

        wbc_b = get_val("白细胞(WBC)"); wbc_a = get_val("白细胞(WBC)")
        neut_b = get_val("中性粒细胞"); neut_a = get_val("中性粒细胞")
        crp_b = get_val("CRP"); crp_a = get_val("CRP")
        pct_b = get_val("PCT"); pct_a = get_val("PCT")
        scr_b = get_val("肌酐"); scr_a = get_val("肌酐")
        egfr_b = get_val("eGFR"); egfr_a = get_val("eGFR")
        alt_b = get_val("丙氨酸氨基转移酶ALT"); alt_a = get_val("丙氨酸氨基转移酶ALT")
        tbil_b = get_val("总胆红素"); tbil_a = get_val("总胆红素")

        # ---- 2.1 肾功能（10分） ----
        with st.expander("肾功能（10分）", expanded=True):
            renal_indicators = [
                ("肌酐", "肌酐", "μmol/L"),
                ("eGFR", "肾小球滤过率", "mL/min/1.73m²"),
                ("尿酸", "尿酸", "μmol/L"),
                ("尿素", "尿素", "mmol/L"),
            ]
            renal_table = []
            for col_name, disp_name, unit in renal_indicators:
                b = rec_before.get(col_name)
                a = rec_after.get(col_name)
                str_b = f"{float(b):.2f}" if b is not None and b != "" else "—"
                str_a = f"{float(a):.2f}" if a is not None and a != "" else "—"
                trend = ""
                try:
                    if float(b) > float(a): trend = "↓"
                    elif float(b) < float(a): trend = "↑"
                    else: trend = "→"
                except: trend = "—"
                renal_table.append({"指标": f"{disp_name}({unit})", "用药前": str_b, "用药后": str_a, "趋势": trend})
            st.table(pd.DataFrame(renal_table))
            renal_score = 10
            renal_desc = []
            if scr_b is not None and scr_a is not None:
                if scr_a > scr_b * 1.2:
                    renal_score -= 5; renal_desc.append("肌酐升高>20%")
                elif scr_a < scr_b * 0.8:
                    renal_score -= 2
            if egfr_b is not None and egfr_a is not None:
                if egfr_a < egfr_b * 0.8:
                    renal_score -= 3; renal_desc.append("eGFR下降>20%")
            if not renal_desc: renal_desc.append("肾功能稳定")
            st.write(f"**自动评分：{max(0, renal_score)}/10 分** — {'；'.join(renal_desc)}")

        # ---- 2.2 肝功能（6分） ----
        with st.expander("肝功能（6分）", expanded=False):
            liver_indicators = [
                ("总胆红素", "总胆红素", "μmol/L"), ("直接胆红素", "直接胆红素", "μmol/L"), ("间接胆红素", "间接胆红素", "μmol/L"),
                ("总蛋白", "总蛋白", "g/L"), ("白蛋白", "白蛋白", "g/L"), ("球蛋白", "球蛋白", "g/L"),
                ("丙氨酸氨基转移酶ALT", "ALT", "U/L"), ("AST", "AST", "U/L"), ("AST:ALT", "AST/ALT", ""),
                ("碱性磷酸酶", "ALP", "U/L"), ("谷氨酰氨基转移酶", "GGT", "U/L"),
            ]
            liver_table = []
            for col_name, disp_name, unit in liver_indicators:
                b = rec_before.get(col_name); a = rec_after.get(col_name)
                str_b = f"{float(b):.2f}" if b is not None and b != "" else "—"
                str_a = f"{float(a):.2f}" if a is not None and a != "" else "—"
                trend = ""
                try:
                    if float(b) > float(a): trend = "↓"
                    elif float(b) < float(a): trend = "↑"
                    else: trend = "→"
                except: trend = "—"
                liver_table.append({"指标": f"{disp_name}({unit})", "用药前": str_b, "用药后": str_a, "趋势": trend})
            st.table(pd.DataFrame(liver_table))
            liver_score = 6; liver_desc = []
            if alt_b is not None and alt_a is not None:
                if alt_a > alt_b * 1.5: liver_score -= 3; liver_desc.append("ALT显著升高")
                elif alt_a < alt_b * 0.7: liver_score += 1
            if tbil_b is not None and tbil_a is not None:
                if tbil_a > tbil_b * 1.5: liver_score -= 2; liver_desc.append("总胆红素显著升高")
                elif tbil_a < tbil_b * 0.7: liver_score += 1
            liver_score = max(0, min(6, liver_score))
            if not liver_desc: liver_desc.append("肝功能大致稳定")
            st.write(f"**自动评分：{liver_score}/6 分** — {'；'.join(liver_desc)}")

        # ---- 2.3 血常规与感染指标（9分） ----
        with st.expander("血常规与感染指标（9分）", expanded=False):
            hem_indicators = [
                ("白细胞(WBC)", "白细胞", "×10⁹/L"), ("中性粒细胞", "中性粒细胞", "%"), ("血小板计数", "血小板", "×10⁹/L"),
                ("CRP", "C反应蛋白", "mg/L"), ("血清淀粉样C蛋白SAA", "SAA", "mg/L"), ("PCT", "降钙素原", "ng/mL"),
            ]
            hem_table = []
            for col_name, disp_name, unit in hem_indicators:
                b = rec_before.get(col_name); a = rec_after.get(col_name)
                str_b = f"{float(b):.2f}" if b is not None and b != "" else "—"
                str_a = f"{float(a):.2f}" if a is not None and a != "" else "—"
                trend = ""
                try:
                    if float(b) > float(a): trend = "↓"
                    elif float(b) < float(a): trend = "↑"
                    else: trend = "→"
                except: trend = "—"
                hem_table.append({"指标": f"{disp_name}({unit})", "用药前": str_b, "用药后": str_a, "趋势": trend})
            st.table(pd.DataFrame(hem_table))
            improve = 0
            if wbc_b and wbc_a and wbc_a < wbc_b: improve += 1
            if crp_b and crp_a and crp_a < crp_b: improve += 1
            if pct_b and pct_a and pct_a < pct_b: improve += 1
            if improve >= 3: hem_score = 9; hem_desc = "三项感染指标均下降"
            elif improve >= 2: hem_score = 6; hem_desc = "两项感染指标下降"
            elif improve >= 1: hem_score = 4; hem_desc = "仅一项下降"
            else: hem_score = 2; hem_desc = "无改善或恶化"
            st.write(f"**自动评分：{hem_score}/9 分** — {hem_desc}")

        lab_total = max(0, renal_score) + liver_score + hem_score
        st.info(f"**实验室炎症指标总分：{lab_total}/25 分**")

        # ==================== 三、病原学（20分） ====================
        st.subheader("三、病原学（满分 20 分）")
        with st.expander("评分", expanded=False):
            st.caption("致病菌清除（15分）：0分（菌群替换/未清除）| 5分（部分清除）| 10分（大部分清除）| 15分（完全清除）")
            p1 = st.select_slider("致病菌清除", options=[0, 5, 10, 15], value=st.session_state.efficacy_scores.get("致病菌清除", 0), key="p1")
            st.caption("药敏结果（5分）：0分（Resistant）| 2分（中介）| 5分（敏感）")
            p2 = st.select_slider("药敏结果", options=[0, 2, 5], value=st.session_state.efficacy_scores.get("药敏结果", 0), key="p2")
            st.session_state.efficacy_scores["致病菌清除"] = p1
            st.session_state.efficacy_scores["药敏结果"] = p2

        # ==================== 四、影像学（15分） ====================
        st.subheader("四、影像学（满分 15 分）")
        with st.expander("评分", expanded=False):
            st.caption("病灶变化（10分）：0分（无变化/扩大）| 3分（缩小30%-69%）| 6分（缩小≥70%）| 10分（基本吸收）")
            i1 = st.select_slider("病灶变化", options=[0, 3, 6, 10], value=st.session_state.efficacy_scores.get("病灶变化", 0), key="i1")
            st.caption("脓肿/积液（5分）：0分（无减少/增多）| 2分（明显减少）| 5分（完全吸收）")
            i2 = st.select_slider("脓肿/积液", options=[0, 2, 5], value=st.session_state.efficacy_scores.get("脓肿积液", 0), key="i2")
            st.session_state.efficacy_scores["病灶变化"] = i1
            st.session_state.efficacy_scores["脓肿积液"] = i2

        # ==================== 五、不良反应（10分） ====================
        st.subheader("五、不良反应（满分 10 分）")
        with st.expander("评分", expanded=False):
            st.caption("不良反应（10分）：0分（严重，需停药）| 2分（中度，调剂量）| 5分（轻微，无需停药）| 10分（无不良反应）")
            a1 = st.select_slider("不良反应", options=[0, 2, 5, 10], value=st.session_state.efficacy_scores.get("不良反应", 0), key="a1")
            st.session_state.efficacy_scores["不良反应"] = a1

        # ==================== 六、扣分项（累计≤10分） ====================
        st.subheader("六、扣分项（累计≤10分）")
        with st.expander("意识异常额外扣分 (0-5分)", expanded=False):
            conscious_penalty = st.select_slider(
                "意识状态 (清醒0分/嗜睡2分/昏迷5分)",
                options=[0, 2, 5],
                value=st.session_state.efficacy_scores.get("意识扣分", 0),
                key="conscious_penalty"
            )
            st.session_state.efficacy_scores["意识扣分"] = conscious_penalty

        with st.expander("特殊人群补充扣分 (累计≤10分)", expanded=False):
            renal_penalty = st.select_slider(
                "肾功能不全 (3-5分)",
                options=[0, 3, 4, 5],
                value=st.session_state.efficacy_scores.get("肾功扣分", 0),
                key="renal_penalty"
            )
            icu_penalty = st.select_slider(
                "ICUSevere (5分)",
                options=[0, 5],
                value=st.session_state.efficacy_scores.get("ICU扣分", 0),
                key="icu_penalty"
            )
            immune_penalty = st.select_slider(
                "免疫低下 (3-5分)",
                options=[0, 3, 4, 5],
                value=st.session_state.efficacy_scores.get("免疫扣分", 0),
                key="immune_penalty"
            )
            st.session_state.efficacy_scores["肾功扣分"] = renal_penalty
            st.session_state.efficacy_scores["ICU扣分"] = icu_penalty
            st.session_state.efficacy_scores["免疫扣分"] = immune_penalty

        total_penalty = conscious_penalty + renal_penalty + icu_penalty + immune_penalty
        total_penalty = min(total_penalty, 10)
        st.info(f"**扣分项合计：{total_penalty}/10 分**")

        # ==================== 总分显示 ====================
        total_score = s1 + s2 + hr + bp + temp + lab_total + p1 + p2 + i1 + i2 + a1 - total_penalty
        total_score = max(0, total_score)
        st.markdown("---")
        st.subheader(f"📊 疗效评估总分：{total_score}/100 分")

        if total_score >= 90:
            疗效结局 = "痊愈"
            st.success(f"🎉 痊愈（≥90分）")
        elif total_score >= 70:
            疗效结局 = "显效"
            st.info(f"👍 显效（70-89分）")
        elif total_score >= 50:
            疗效结局 = "有效"
            st.warning(f"📌 有效（50-69分）")
        else:
            疗效结局 = "无效"
            st.error(f"⚠️ 无效（<50分）")

        st.caption("判定标准：≥90分为痊愈，70-89分为显效，50-69分为有效，<50分为无效")

        # ==================== 保存到数据库 ====================
        if st.button("💾 保存评估结果到数据库"):
            eval_record = {
                "编号": 患者编号,
                "日期": 用药后日期,
                "疗效总分": total_score,
                "疗效结局": 疗效结局,
                "record_id": str(uuid.uuid4())
            }
            st.session_state.patient_db.append(eval_record)
            st.success("评估结果已保存至数据库！")

        if st.button("🔄 重新选择日期"):
            st.session_state.eval_started = False
            st.rerun()

    st.stop()
# ----------------------------- 统计分析模块（完整版） -----------------------------
if st.session_state.show_stats:
    st.subheader("📊 全指标统计分析中心")
    
    # ===== Excel模板下载 =====
    st.markdown("### 📥 下载标准导入模板")
    template_cols = [
        "编号", "日期", "年龄", "性别", "身高", "体重", "体温", "心率", "血压",
        "感染部位", "感染程度", "病原菌列表", "MIC列表",
        "肌酐", "eGFR", "尿酸", "尿素",
        "总胆红素", "直接胆红素", "间接胆红素", "总蛋白", "白蛋白", "球蛋白",
        "丙氨酸氨基转移酶ALT", "AST", "AST:ALT", "碱性磷酸酶", "谷氨酰氨基转移酶",
        "白细胞(WBC)", "中性粒细胞", "血小板计数", "CRP", "血清淀粉样C蛋白SAA", "PCT",
        "给药前3h浓度", "给药前0.5h浓度", "剂量", "间隔", "输注时间", "结论类型"
    ]
    template_df = pd.DataFrame(columns=template_cols)
    template_df.loc[0] = [
        "P001", "2026-04-22", 65, "男", 170, 70, 36.8, 80, "120/80",
        "肺部感染", "严重", "肺炎克雷伯菌,大肠埃希菌", "2,4",
        88, 85, 320, 5.6, 12.5, 4.2, 8.3, 72, 42, 30,
        25, 28, 1.12, 65, 30, 7.5, 65, 220, 5.2, 8.1, 0.05,
        12.3, 4.5, 1.0, 8, 0.5, "血药浓度达标"
    ]
    csv = template_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="⬇️ 下载标准导入模板 (CSV格式)",
        data=csv,
        file_name="美罗培南导入模板.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption("请严格按照模板列名填写数据，日期格式为YYYY-MM-DD。")

    st.divider()
    st.markdown("### 📂 导入外部病例数据（Excel）")
    
    import_mode = st.radio(
        "选择导入模式",
        ["完整导入（每行作为新记录）", "按编号+日期合并更新"],
        horizontal=True,
        help="按编号+日期合并：若数据库中已存在相同编号且同一天的记录，则用新数据中的非空值更新；否则新增记录。"
    )
    
    uploaded_files = st.file_uploader(
        "上传一个或多个Excel/CSV文件（支持多工作表）",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_dfs = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            try:
                if file_name.endswith('.csv'):
                    df_temp = pd.read_csv(uploaded_file)
                    all_dfs.append(df_temp)
                else:
                    sheets = pd.read_excel(uploaded_file, sheet_name=None)
                    for sheet_name, df_sheet in sheets.items():
                        all_dfs.append(df_sheet)
            except Exception as e:
                st.error(f"读取文件 {file_name} 失败：{e}")
        
        if not all_dfs:
            st.warning("没有成功读取到任何数据")
        else:
            try:
                df_import = pd.concat(all_dfs, ignore_index=True, join='outer')
                if "日期" in df_import.columns:
                    df_import["日期"] = pd.to_datetime(df_import["日期"], errors='coerce')
                    df_import["日期"] = df_import["日期"].dt.strftime('%Y-%m-%d')
                st.success(f"成功读取 {len(df_import)} 行数据（来自 {len(uploaded_files)} 个文件）")
                st.dataframe(df_import.head(10), width='stretch')
                st.caption("展示前10行预览，请确认数据无误后点击导入按钮")
                
                if st.button("确认导入到数据库"):
                    if import_mode == "完整导入（每行作为新记录）":
                        for _, row in df_import.iterrows():
                            record = row.to_dict()
                            record.setdefault("病原菌列表", [])
                            record.setdefault("MIC列表", [1])
                            record.setdefault("结论类型", "未知")
                            record.setdefault("编号", "")
                            record.setdefault("日期", "")
                            if "病号" in record:
                                record["编号"] = record.pop("病号")
                            if "感染部位" in record and isinstance(record["感染部位"], str):
                                record["感染部位"] = [v.strip() for v in record["感染部位"].split(',') if v.strip()]
                            if "日期" in record:
                                record["日期"] = str(record["日期"]).strip()
                            record["record_id"] = str(uuid.uuid4())
                            st.session_state.patient_db.append(record)
                            # 同步到全局数据库
                            if 'global_db' in dir() and st.session_state.user_role == "user":
                                global_db.add_record(record)
                        st.success(f"已导入 {len(df_import)} 条记录")
                        st.rerun()
                        
                    else:  # 按编号+日期合并
                        if "编号" not in df_import.columns:
                            st.error("Excel中缺少'编号'列，无法合并。")
                        elif "日期" not in df_import.columns:
                            st.error("Excel中缺少'日期'列，无法按日期合并。")
                        else:
                            updated_count = 0
                            new_count = 0
                            
                            for _, new_row in df_import.iterrows():
                                patient_id = str(new_row["编号"]).strip() if pd.notna(new_row["编号"]) else ""
                                date_val = str(new_row["日期"]).strip() if pd.notna(new_row["日期"]) else ""
                                
                                if not patient_id or not date_val:
                                    record = new_row.to_dict()
                                    record.setdefault("病原菌列表", [])
                                    record.setdefault("MIC列表", [1])
                                    record.setdefault("结论类型", "未知")
                                    if "感染部位" in record and isinstance(record["感染部位"], str):
                                        record["感染部位"] = [v.strip() for v in record["感染部位"].split(',') if v.strip()]
                                    if "日期" in record:
                                        record["日期"] = str(record["日期"]).strip()
                                    record["record_id"] = str(uuid.uuid4())
                                    st.session_state.patient_db.append(record)
                                    if 'global_db' in dir() and st.session_state.user_role == "user":
                                        global_db.add_record(record)
                                    new_count += 1
                                    continue
                                
                                matched_indices = []
                                for i, rec in enumerate(st.session_state.patient_db):
                                    if (str(rec.get("编号", "")).strip() == patient_id and 
                                        str(rec.get("日期", "")).strip() == date_val):
                                        matched_indices.append(i)
                                
                                if matched_indices:
                                    idx = matched_indices[0]
                                    existing_rec = st.session_state.patient_db[idx]
                                    for key, value in new_row.items():
                                        if pd.notna(value) and str(value).strip() not in ["", "无", "nan", "NaN"]:
                                            if key == "病原菌列表" or key == "MIC列表":
                                                if isinstance(value, str):
                                                    items = [v.strip() for v in value.split(',') if v.strip()]
                                                    if items:
                                                        existing_rec[key] = items
                                            elif key == "感染部位":
                                                if isinstance(value, str):
                                                    items = [v.strip() for v in value.split(',') if v.strip()]
                                                    existing_rec[key] = items if items else value
                                                elif isinstance(value, list):
                                                    existing_rec[key] = value
                                            elif key == "日期":
                                                existing_rec[key] = str(value).strip()
                                            else:
                                                existing_rec[key] = value
                                    updated_count += 1
                                else:
                                    record = new_row.to_dict()
                                    record.setdefault("病原菌列表", [])
                                    record.setdefault("MIC列表", [1])
                                    record.setdefault("结论类型", "未知")
                                    if "感染部位" in record and isinstance(record["感染部位"], str):
                                        record["感染部位"] = [v.strip() for v in record["感染部位"].split(',') if v.strip()]
                                    if "日期" in record:
                                        record["日期"] = str(record["日期"]).strip()
                                    record["record_id"] = str(uuid.uuid4())
                                    st.session_state.patient_db.append(record)
                                    if 'global_db' in dir() and st.session_state.user_role == "user":
                                        global_db.add_record(record)
                                    new_count += 1
                            
                            st.success(f"导入完成：新增 {new_count} 条记录，更新 {updated_count} 条记录")
                            st.rerun()
            except Exception as e:
                st.error(f"导入失败：{e}")

    st.divider()
    # 根据角色获取数据源
    df_raw = pd.DataFrame(st.session_state.patient_db)
    if "日期" in df_raw.columns:
        df_raw["日期"] = df_raw["日期"].astype(str).str.strip()
    st.markdown(f"**总纳入记录数：{len(df_raw)} 例**")

    if df_raw.empty:
        st.warning("暂无患者数据")
    else:
        st.markdown("### 数据查看方式")
        view_mode = st.radio(
            "选择展示模式",
            ["原始记录（逐条）", "按编号分组（显示各日期记录）"],
            horizontal=True
        )

        # 导出按钮通用函数
        def show_export_button(dataframe, filename="患者数据导出.xlsx"):
            # 将列表列转换为字符串以便导出
            df_export = dataframe.copy()
            for col in df_export.columns:
                if df_export[col].apply(lambda x: isinstance(x, list)).any():
                    df_export[col] = df_export[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x))
            # 生成Excel文件
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)
            st.download_button(
                label="📥 导出为Excel",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # 安全转换函数
        def safe_convert(x):
            if isinstance(x, list):
                return ", ".join(map(str, x))
            return str(x) if x is not None else ""

        df_display = df_raw.copy()
        list_cols = ["病原菌列表", "MIC列表"]
        for col in list_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(safe_convert)

        if view_mode == "按编号分组（显示各日期记录）":
            if "编号" not in df_display.columns:
                st.warning("数据库中没有'编号'字段，无法分组。")
            else:
                # 导出按钮（导出全部数据，不分组）
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("")
                with col2:
                    show_export_button(df_raw)
                st.divider()
                
                grouped_by_id = df_display.groupby("编号")
                for patient_id, group_df in grouped_by_id:
                    st.subheader(f"编号：{patient_id} （共 {len(group_df)} 条记录）")
                    group_cols = [c for c in group_df.columns if c != "record_id"]
                    priority = ["编号", "日期", "年龄", "性别", "感染部位", "感染程度",
                                "剂量", "间隔", "结论类型"]
                    ordered = [c for c in priority if c in group_cols]
                    for c in group_cols:
                        if c not in ordered:
                            ordered.append(c)
                    group_df_ordered = group_df[ordered].copy()
                    group_df_ordered.insert(0, "序号", range(1, len(group_df_ordered) + 1))
                    group_df_show = group_df_ordered.copy()
                    for c in group_df_show.columns:
                        group_df_show[c] = group_df_show[c].apply(safe_convert)
                    st.dataframe(group_df_show, width='stretch')
                    st.divider()
        else:
            st.subheader("📋 原始病例记录列表")
            # 导出按钮
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("")
            with col2:
                show_export_button(df_raw)
            st.divider()
            
            display_cols = [c for c in df_display.columns if c != "record_id"]
            priority = ["编号", "日期"]
            for p in reversed(priority):
                if p in display_cols:
                    display_cols.remove(p)
                    display_cols.insert(0, p)
            df_display.insert(0, "序号", range(1, len(df_display)+1))
            display_cols = ["序号"] + display_cols
            df_show = df_display[display_cols].copy()
            for c in df_show.columns:
                df_show[c] = df_show[c].apply(safe_convert)

            event = st.dataframe(
                df_show,
                width='stretch',
                key="main_dataframe",
                selection_mode="multi-row",
                on_select="rerun"
            )
            selected_indices = event.selection.get("rows", [])
            selected_count = len(selected_indices)
            st.caption(f"已选中 {selected_count} 条记录")

            if st.button("🗑️ 删除选中记录", use_container_width=True):
                if selected_count == 0:
                    st.warning("请先勾选要删除的记录")
                else:
                    keep_indices = [i for i in range(len(df_raw)) if i not in selected_indices]
                    st.session_state.patient_db = [st.session_state.patient_db[i] for i in keep_indices]
                    if st.session_state.get("user_role") == "admin" and 'global_db' in dir():
                        global_db.data = st.session_state.patient_db
                        global_db.save()
                    st.success(f"已删除 {selected_count} 条记录")
                    st.rerun()

        st.divider()

        # ==================== 分析模式切换 ====================
        analysis_mode = st.radio(
            "选择分析模式",
            ["📊 常规指标分析", "📈 协变量分析（对血药浓度达标的影响）"],
            horizontal=True
        )

        if analysis_mode == "📊 常规指标分析":
            st.subheader("📈 统计指标分析")
            if df_raw.empty:
                st.warning("暂无数据")
            else:
                # 智能列名匹配函数
                def find_column(df, target_col):
                    if target_col in df.columns:
                        return target_col
                    aliases = {
                        "白细胞(WBC)": ["白细胞", "WBC"],
                        "丙氨酸氨基转移酶ALT": ["ALT", "丙氨酸氨基转移酶"],
                        "血清淀粉样C蛋白SAA": ["SAA", "血清淀粉样C蛋白"],
                        "eGFR": ["肾小球滤过率", "GFR"],
                        "CRP": ["C反应蛋白"],
                        "PCT": ["降钙素原"],
                        "AST:ALT": ["AST/ALT", "AST_ALT"],
                        "病原菌列表": ["病原菌"],
                    }
                    for alias in aliases.get(target_col, []):
                        if alias in df.columns:
                            return alias
                    return None

                # 29个指标映射
                indicator_map = {
                    "年龄": "年龄",
                    "性别": "性别",
                    "身高": "身高",
                    "体重": "体重",
                    "心率": "心率",
                    "体温": "体温",
                    "感染部位": "感染部位",
                    "感染程度": "感染程度",
                    "病原菌": "病原菌列表",
                    "肌酐": "肌酐",
                    "肾小球滤过率": "eGFR",
                    "尿酸": "尿酸",
                    "尿素": "尿素",
                    "总蛋白": "总蛋白",
                    "白蛋白": "白蛋白",
                    "球蛋白": "球蛋白",
                    "总胆红素": "总胆红素",
                    "间接胆红素": "间接胆红素",
                    "直接胆红素": "直接胆红素",
                    "丙氨酸氨基转移酶": "丙氨酸氨基转移酶ALT",
                    "碱性磷酸酶": "碱性磷酸酶",
                    "AST:ALT": "AST:ALT",
                    "谷氨酰氨基转移酶": "谷氨酰氨基转移酶",
                    "血小板计数": "血小板计数",
                    "白细胞计数": "白细胞(WBC)",
                    "中性粒细胞": "中性粒细胞",
                    "C反应蛋白": "CRP",
                    "降钙素原": "PCT",
                    "血清淀粉样C蛋白": "血清淀粉样C蛋白SAA",
                    "结果解释": "解释存在",
                    "结论类型": "结论类型",
                    "滴注速度": "滴注速度",
                }

                # 横坐标标签映射（英文）
                xlabel_map = {
                    "年龄": "Age (years)",
                    "身高": "Height (cm)",
                    "体重": "Weight (kg)",
                    "心率": "Heart Rate (/min)",
                    "肌酐": "Creatinine (μmol/L)",
                    "eGFR": "eGFR (mL/min/1.73m²)",
                    "尿酸": "Uric Acid (μmol/L)",
                    "尿素": "Urea (mmol/L)",
                    "总蛋白": "Total Protein (g/L)",
                    "白蛋白": "Albumin (g/L)",
                    "球蛋白": "Globulin (g/L)",
                    "总胆红素": "Total Bilirubin (μmol/L)",
                    "直接胆红素": "Direct Bilirubin (μmol/L)",
                    "间接胆红素": "Indirect Bilirubin (μmol/L)",
                    "丙氨酸氨基转移酶ALT": "ALT (U/L)",
                    "碱性磷酸酶": "ALP (U/L)",
                    "AST:ALT": "AST/ALT",
                    "谷氨酰氨基转移酶": "GGT (U/L)",
                    "血小板计数": "Platelets (×10⁹/L)",
                    "白细胞(WBC)": "WBC (×10⁹/L)",
                    "中性粒细胞": "Neutrophils (%)",
                    "CRP": "CRP (mg/L)",
                    "PCT": "PCT (ng/mL)",
                    "血清淀粉样C蛋白SAA": "SAA (mg/L)",
                    "滴注速度": "Infusion Rate (mL/min)",
                }

                all_display = list(indicator_map.keys())
                col1, col2 = st.columns(2)
                with col1:
                    target_display = st.selectbox("选择分析指标", all_display)
                with col2:
                    chart_type = st.selectbox("图表类型", ["自动", "频数分布图", "饼图", "折线图"])

                target_col = find_column(df_raw, indicator_map[target_display])
                if target_col is None and target_display != "结果解释":
                    st.warning(f"数据库中未找到 '{indicator_map[target_display]}' 对应的列，请检查数据。")
                else:
                    categorical_fields = ["性别", "体温", "感染部位", "感染程度", "病原菌", "结论类型", "结果解释"]

                    if target_display in categorical_fields:
                        # 分类变量处理
                        if target_display == "病原菌":
                            all_pathogens = []
                            for val in df_raw[target_col].dropna():
                                if isinstance(val, list):
                                    all_pathogens.extend(val)
                                elif isinstance(val, str) and val.strip():
                                    items = [v.strip() for v in val.split(',') if v.strip()]
                                    all_pathogens.extend(items)
                            series = pd.Series(all_pathogens)
                        elif target_display == "感染部位":
                            all_sites = []
                            for val in df_raw[target_col].dropna():
                                if isinstance(val, list):
                                    all_sites.extend(val)
                                elif isinstance(val, str) and val.strip():
                                    items = [v.strip() for v in val.split(',') if v.strip()]
                                    all_sites.extend(items)
                            series = pd.Series(all_sites)
                        elif target_display == "结果解释":
                            has_expl = df_raw.apply(lambda row: bool(row.get("建议", "")) or bool(row.get("解释", "")), axis=1)
                            series = has_expl.map({True: "Has Explanation", False: "No Explanation"})
                        else:
                            series = df_raw[target_col].dropna().astype(str)
                        
                        if len(series) == 0:
                            st.warning("该字段无有效数据")
                        else:
                            cnt = series.value_counts()
                            fig, ax = plt.subplots(figsize=(8, 5))
                            if chart_type in ["自动", "频数分布图"]:
                                bars = ax.bar(cnt.index, cnt.values, color="#2C7DDB", alpha=0.7)
                                ax.set_xlabel(target_display)
                                ax.set_ylabel("Count")
                                for bar, value in zip(bars, cnt.values):
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                            str(value), ha='center', va='bottom', fontsize=10)
                            elif chart_type == "饼图":
                                cnt.plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90)
                                ax.set_ylabel("")
                            elif chart_type == "折线图":
                                cnt_sorted = cnt.sort_index()
                                ax.plot(cnt_sorted.index, cnt_sorted.values, marker='o', color="#2C7DDB")
                                ax.set_xlabel(target_display)
                                ax.set_ylabel("Count")
                                for x, y in zip(cnt_sorted.index, cnt_sorted.values):
                                    ax.text(x, y + 0.1, str(y), ha='center', va='bottom', fontsize=9)
                            ax.set_title(f"{target_display} Distribution")
                            st.pyplot(fig)
                            st.dataframe(cnt.rename("Count"), width='stretch')

                    else:
                        # 数值变量处理
                        s = pd.to_numeric(df_raw[target_col], errors='coerce').dropna()
                        if len(s) == 0:
                            st.warning("该字段无有效数值数据")
                        else:
                            if target_display == "年龄":
                                bins = np.arange(0, 101, 10)
                            elif target_display == "身高":
                                bins = np.arange(100, 221, 5)
                            elif target_display == "体重":
                                bins = np.arange(30, 151, 5)
                            else:
                                min_val = max(0, np.floor(s.min()))
                                max_val = np.ceil(s.max())
                                if max_val > min_val:
                                    bins = np.linspace(min_val, max_val, 11)
                                else:
                                    bins = np.array([min_val, min_val+1])

                            counts, bin_edges = np.histogram(s, bins=bins)
                            xlabel = xlabel_map.get(target_col, target_display)

                            fig, ax = plt.subplots(figsize=(8, 5))
                            if chart_type == "自动" or chart_type == "频数分布图":
                                bars = ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                                              align="edge", color="#2C7DDB", alpha=0.7, edgecolor='white')
                                ax.set_xlabel(xlabel)
                                ax.set_ylabel("Count")
                                for bar, count in zip(bars, counts):
                                    if count > 0:
                                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                                str(count), ha='center', va='bottom', fontsize=9)
                            elif chart_type == "饼图":
                                labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]
                                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.set_ylabel("")
                            elif chart_type == "折线图":
                                sorted_s = np.sort(s)
                                y = np.arange(1, len(sorted_s)+1) / len(sorted_s)
                                ax.plot(sorted_s, y, marker='.', linestyle='-', color="#2C7DDB")
                                ax.set_xlabel(xlabel)
                                ax.set_ylabel("Cumulative Proportion")
                            ax.set_title(f"{target_display} Distribution")
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown("### 描述性统计")
                            st.dataframe(s.describe().round(2), width='stretch')

        else:  # 协变量分析模式
            st.subheader("📈 协变量对血药浓度达标的影响分析")
            if df_raw.empty:
                st.warning("暂无数据")
            else:
                # 中→英特征名映射
                feature_name_map = {
                    "年龄": "Age",
                    "性别": "Sex",
                    "身高": "Height",
                    "体重": "Weight",
                    "心率": "Heart Rate",
                    "体温": "Temperature",
                    "感染部位": "Infection Site",
                    "感染程度": "Severity",
                    "病原菌列表": "Pathogen",
                    "肌酐": "Creatinine",
                    "eGFR": "eGFR",
                    "尿酸": "Uric Acid",
                    "尿素": "Urea",
                    "总蛋白": "Total Protein",
                    "白蛋白": "Albumin",
                    "球蛋白": "Globulin",
                    "总胆红素": "Total Bilirubin",
                    "间接胆红素": "Indirect Bilirubin",
                    "直接胆红素": "Direct Bilirubin",
                    "丙氨酸氨基转移酶ALT": "ALT",
                    "碱性磷酸酶": "ALP",
                    "AST:ALT": "AST/ALT",
                    "谷氨酰氨基转移酶": "GGT",
                    "血小板计数": "Platelets",
                    "白细胞(WBC)": "WBC",
                    "中性粒细胞": "Neutrophils",
                    "CRP": "CRP",
                    "PCT": "PCT",
                    "血清淀粉样C蛋白SAA": "SAA",
                    "滴注速度": "Infusion Rate",
                }

                candidate_cols = list(feature_name_map.keys())
                num_cols = []
                for col in candidate_cols:
                    if col in df_raw.columns:
                        try:
                            pd.to_numeric(df_raw[col], errors='raise')
                            num_cols.append(col)
                        except:
                            continue
                
                if not num_cols:
                    st.warning("没有可分析的数值协变量，请先录入数据。")
                else:
                    outcome_type = st.selectbox("结局定义", ["达标 vs 未达标", "三分类（过高/达标/过低）"])
                    if outcome_type == "达标 vs 未达标":
                        y = (df_raw["结论类型"] == "血药浓度达标").astype(int)
                    else:
                        y = df_raw["结论类型"].astype(str)

                    with st.expander("🔍 随机森林特征重要性（全部29个指标）", expanded=True):
                        try:
                            from sklearn.ensemble import RandomForestClassifier
                            all_29_cols = list(feature_name_map.keys())
                            existing_cols = [c for c in all_29_cols if c in df_raw.columns]
                            if existing_cols:
                                X = df_raw[existing_cols].copy()
                                cat_cols = ["性别", "感染部位", "感染程度", "病原菌列表"]
                                for col in cat_cols:
                                    if col in X.columns:
                                        le = LabelEncoder()
                                        X[col] = X[col].apply(lambda x: str(x) if not isinstance(x, list) else ",".join(map(str, x)))
                                        X[col] = le.fit_transform(X[col].astype(str))
                                X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                                mask = y.notna()
                                X = X[mask]
                                y_clean = y[mask]
                                if len(X) > 5:
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                    model.fit(X, y_clean)
                                    importances = model.feature_importances_
                                    eng_features = [feature_name_map.get(c, c) for c in existing_cols]
                                    imp_df = pd.DataFrame({"Feature": eng_features, "Importance": importances}).sort_values("Importance", ascending=False)
                                    st.dataframe(imp_df, width='stretch', height=400)
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.barh(imp_df["Feature"][:15], imp_df["Importance"][:15])
                                    ax.set_xlabel("Importance")
                                    ax.set_title("Feature Importance for Target Attainment")
                                    ax.spines['top'].set_visible(False)
                                    ax.spines['right'].set_visible(False)
                                    st.pyplot(fig)
                                else:
                                    st.info("有效数据不足")
                        except Exception as e:
                            st.error(f"随机森林计算出错：{e}")

                    st.markdown("### Single Covariate Group Comparison")
                    selected_cov_zh = st.selectbox(
                        "Select Covariate",
                        num_cols,
                        format_func=lambda x: feature_name_map.get(x, x)
                    )
                    selected_cov_en = feature_name_map.get(selected_cov_zh, selected_cov_zh)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df_plot = df_raw[[selected_cov_zh, "结论类型"]].copy()
                    df_plot[selected_cov_zh] = pd.to_numeric(df_plot[selected_cov_zh], errors='coerce')
                    df_plot.dropna(subset=[selected_cov_zh, "结论类型"], inplace=True)
                    if outcome_type == "达标 vs 未达标":
                        df_plot["Group"] = (df_plot["结论类型"] == "血药浓度达标").map({True: "Attained", False: "Not Attained"})
                    else:
                        df_plot["Group"] = df_plot["结论类型"]
                    groups = sorted(df_plot["Group"].unique())
                    data_to_plot = [df_plot[df_plot["Group"] == g][selected_cov_zh].values for g in groups]
                    ax.boxplot(data_to_plot, labels=groups)
                    ax.set_xlabel("Group")
                    ax.set_ylabel(selected_cov_en)
                    ax.set_title(f"{selected_cov_en} by Group")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)

                    st.markdown("### Group Descriptive Statistics")
                    stats = df_plot.groupby("Group")[selected_cov_zh].agg(["count", "mean", "std", "min", "max"]).round(2)
                    st.dataframe(stats, width='stretch', height=200)
    st.stop()

# ----------------------------- 模型训练与持续学习模块（完整版） -----------------------------
if st.session_state.show_training:
    st.subheader("🤖 模型训练与持续学习")
    if st.button("← 返回录入页面", key="train_back"):
        st.session_state.show_training = False
        st.rerun()

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
    import joblib
    import os

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    st.markdown("训练PK预测模型与血药浓度达标预测模型，使用优化后的随机森林算法。")

    df = pd.DataFrame(shared_db.data if 'shared_db' in dir() else st.session_state.patient_db)
    st.markdown(f"**当前训练数据量：{len(df)} 条记录**")

    if "给药前0.5h浓度" in df.columns and "结论类型" in df.columns:
        valid_mask = df["给药前0.5h浓度"].notna() & df["结论类型"].notna()
        valid_count = valid_mask.sum()
        st.markdown(f"**有效数据量：{valid_count} 条**（同时具有实测谷浓度和结论类型）")
    else:
        st.markdown("**有效数据量：0 条**")

    if not df.empty:
        with st.expander("查看现有训练数据"):
            df_show = df.copy()
            def safe_convert(x):
                if isinstance(x, list):
                    return ", ".join(map(str, x))
                return str(x) if x is not None else ""
            for col in df_show.columns:
                df_show[col] = df_show[col].apply(safe_convert)
            st.dataframe(df_show, width='stretch')

    st.divider()
    st.markdown("### 🧠 训练模型（随机森林优化版）")

    if len(df) < 10:
        st.warning("训练数据不足（至少10条）")
    else:
        df = pd.DataFrame(shared_db.data if 'shared_db' in dir() else st.session_state.patient_db)
        if "给药前0.5h浓度" not in df.columns or "结论类型" not in df.columns:
            st.error("缺少必要列")
        else:
            df_eng = add_engineered_features(df)

            base_feature_cols = [
                "年龄", "身高", "体重", "心率", "体温",
                "感染部位_首位", "感染程度", "病原菌列表", "MIC列表",
                "肌酐", "eGFR", "尿酸", "尿素",
                "总蛋白", "白蛋白", "球蛋白",
                "总胆红素", "直接胆红素", "间接胆红素",
                "丙氨酸氨基转移酶ALT", "碱性磷酸酶", "AST:ALT", "谷氨酰氨基转移酶",
                "血小板计数", "白细胞(WBC)", "中性粒细胞",
                "CRP", "PCT", "血清淀粉样C蛋白SAA",
                "剂量", "间隔", "滴注速度"
            ]
            engineered_cols_candidates = ["BMI", "CrCl", "日总剂量", "剂量_体重", "是否老年",
                                          "CL_pred", "CrCl_x_老年", "是否耐药"]
            engineered_cols = [c for c in engineered_cols_candidates if c in df_eng.columns]
            available_base = [c for c in base_feature_cols if c in df_eng.columns]
            all_feature_cols = available_base + engineered_cols

            cat_cols = ["性别", "感染部位_首位", "感染程度", "病原菌列表"]
            X = df_eng[all_feature_cols].copy()
            for col in cat_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = X[col].astype(str)
                    X[col] = le.fit_transform(X[col])

            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            y_reg = pd.to_numeric(df_eng["给药前0.5h浓度"], errors='coerce')
            y_cls_str = df_eng["结论类型"].astype(str).str.strip().str.replace(" ", "")

            mask_clean = (y_reg >= 0.1) & (y_reg <= 30)
            X = X[mask_clean]
            y_reg = y_reg[mask_clean]
            y_cls_str = y_cls_str[mask_clean]

            mask = y_reg.notna() & y_cls_str.notna() & (y_cls_str != "")
            X = X[mask]
            y_reg = y_reg[mask]
            y_cls_str = y_cls_str[mask]

            st.write("**清洗后结论类型分布：**", y_cls_str.value_counts().to_dict())

            if len(X) < 10:
                st.warning(f"有效完整数据不足（当前{len(X)}条）")
            else:
                le_outcome = LabelEncoder()
                y_cls = le_outcome.fit_transform(y_cls_str)
                classes = le_outcome.classes_
                can_train_cls = len(classes) >= 2

                X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
                    X, y_reg, y_cls, test_size=0.2, random_state=42
                )

                col_tr1, col_tr2 = st.columns(2)
                with col_tr1:
                    if st.button("训练PK预测模型（保存至桌面）", key="train_pk_stable"):
                        with st.spinner("训练中..."):
                            model_reg = RandomForestRegressor(
                                n_estimators=300, max_depth=7, min_samples_split=5,
                                max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1
                            )
                            model_reg.fit(X_train, y_reg_train)
                            pred_reg = model_reg.predict(X_test)

                            mae = mean_absolute_error(y_reg_test, pred_reg)
                            rmse = np.sqrt(mean_squared_error(y_reg_test, pred_reg))
                            r2 = r2_score(y_reg_test, pred_reg)

                            st.success(f"PK模型训练完成！")
                            st.metric("测试集 MAE", f"{mae:.2f} mg/L")
                            st.metric("测试集 RMSE", f"{rmse:.2f} mg/L")
                            st.metric("测试集 R²", f"{r2:.3f}")

                            joblib.dump(model_reg, os.path.join(desktop, "pk_model.pkl"))
                            joblib.dump(all_feature_cols, os.path.join(desktop, "pk_features.pkl"))
                            st.session_state.pk_model = model_reg
                            st.session_state.pk_features = all_feature_cols

                with col_tr2:
                    if can_train_cls:
                        if st.button("训练血药浓度达标预测模型（保存至桌面）", key="train_cls_stable"):
                            with st.spinner("训练中..."):
                                model_cls = RandomForestClassifier(
                                    n_estimators=300, max_depth=7, min_samples_split=5,
                                    max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1
                                )
                                model_cls.fit(X_train, y_cls_train)
                                pred_cls = model_cls.predict(X_test)

                                acc = accuracy_score(y_cls_test, pred_cls)
                                st.success(f"血药浓度达标预测模型训练完成！测试集准确率：{acc:.2%}")
                                st.text("分类报告：\n" + classification_report(y_cls_test, pred_cls, target_names=classes, zero_division=0))

                                fig, ax = plt.subplots(figsize=(4, 3))
                                ConfusionMatrixDisplay.from_predictions(y_cls_test, pred_cls, display_labels=classes, ax=ax, cmap='Blues')
                                st.pyplot(fig)

                                joblib.dump(model_cls, os.path.join(desktop, "outcome_model.pkl"))
                                joblib.dump(le_outcome, os.path.join(desktop, "outcome_encoder.pkl"))
                                st.session_state.outcome_model = model_cls
                                st.session_state.outcome_encoder = le_outcome
                    else:
                        st.warning("❌ 分类目标只有一种类别，无法训练血药浓度达标预测模型。请检查结论类型字段或增加不同疗效的病例。")

                # 特征重要性图表（英文标注）
                feature_name_map = {
                    "年龄": "Age", "身高": "Height", "体重": "Weight", "心率": "Heart Rate",
                    "体温": "Temperature", "感染部位_首位": "Infection Site", "感染程度": "Severity",
                    "病原菌列表": "Pathogen", "MIC列表": "MIC", "肌酐": "Creatinine", "eGFR": "eGFR",
                    "尿酸": "Uric Acid", "尿素": "Urea", "总蛋白": "Total Protein", "白蛋白": "Albumin",
                    "球蛋白": "Globulin", "总胆红素": "Total Bilirubin", "直接胆红素": "Direct Bilirubin",
                    "间接胆红素": "Indirect Bilirubin", "丙氨酸氨基转移酶ALT": "ALT", "碱性磷酸酶": "ALP",
                    "AST:ALT": "AST/ALT", "谷氨酰氨基转移酶": "GGT", "血小板计数": "Platelets",
                    "白细胞(WBC)": "WBC", "中性粒细胞": "Neutrophils", "CRP": "CRP", "PCT": "PCT",
                    "血清淀粉样C蛋白SAA": "SAA", "剂量": "Dose", "间隔": "Interval", "滴注速度": "Infusion Rate",
                    "BMI": "BMI", "CrCl": "CrCl", "日总剂量": "Daily Dose", "剂量_体重": "Dose/Weight",
                    "是否老年": "Elderly", "CL_pred": "CL_pred", "CrCl_x_老年": "CrCl*Elderly", "是否耐药": "Resistant"
                }

                if "pk_model" in st.session_state:
                    with st.expander("View PK Model Feature Importance"):
                        importances = st.session_state.pk_model.feature_importances_
                        features = st.session_state.pk_features
                        eng_features = [feature_name_map.get(f, f) for f in features]
                        imp_df = pd.DataFrame({"Feature": eng_features, "Importance": importances}).sort_values("Importance", ascending=False)
                        threshold = st.slider("Importance threshold", 0.0, float(imp_df["Importance"].max()), 0.02, 0.005, key="pk_threshold")
                        st.dataframe(imp_df, width='stretch')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top = imp_df.head(15)
                        colors = ["#2C7DDB" if imp >= threshold else "#cccccc" for imp in top["Importance"]]
                        ax.barh(top["Feature"], top["Importance"], color=colors)
                        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold {threshold:.3f}')
                        ax.set_xlabel("Importance")
                        ax.set_title("PK Model Feature Importance (Top 15)")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.legend()
                        st.pyplot(fig)

                if "outcome_model" in st.session_state:
                    with st.expander("View Outcome Model Feature Importance"):
                        importances_cls = st.session_state.outcome_model.feature_importances_
                        imp_df_cls = pd.DataFrame({"Feature": eng_features, "Importance": importances_cls}).sort_values("Importance", ascending=False)
                        threshold_cls = st.slider("Importance threshold", 0.0, float(imp_df_cls["Importance"].max()), 0.02, 0.005, key="cls_threshold")
                        st.dataframe(imp_df_cls, width='stretch')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_cls = imp_df_cls.head(15)
                        colors_cls = ["#2C7DDB" if imp >= threshold_cls else "#cccccc" for imp in top_cls["Importance"]]
                        ax.barh(top_cls["Feature"], top_cls["Importance"], color=colors_cls)
                        ax.axvline(threshold_cls, color='red', linestyle='--', label=f'Threshold {threshold_cls:.3f}')
                        ax.set_xlabel("Importance")
                        ax.set_title("Efficacy Model Feature Importance (Top 15)")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.legend()
                        st.pyplot(fig)

    st.stop()
# ======================== 主页面录入 ========================
if not any([st.session_state.show_stats, st.session_state.show_bayes, st.session_state.show_validation,
            st.session_state.show_efficacy, st.session_state.show_montecarlo, st.session_state.show_lasso,
            st.session_state.show_training]):

    st.markdown("## 一、基础信息")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        编号 = st.text_input("编号（唯一标识）", placeholder="例如：P001")
    with col2:
        日期 = st.date_input("日期", value=dt_date.today())
    with col3:
        年龄 = st.text_input("年龄（岁）", placeholder="无/数值")
    with col4:
        体重 = st.text_input("体重（kg）", placeholder="无/数值")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        身高 = st.text_input("身高（cm）", placeholder="无/数值")
    with col6:
        性别 = st.selectbox("性别", ["请选择", "男", "女"])
    with col7:
        心率 = st.text_input("心率（次/分）", placeholder="无/数值")
    with col8:
        体温 = st.text_input("体温（℃）", placeholder="无/数值")

    col9, col10 = st.columns(2)
    with col9:
        血压 = st.text_input("血压（mmHg）", placeholder="例：120/80")
    with col10:
        pass

    # ---------- 感染信息 ----------
    st.markdown("## 二、感染信息")
    if "infection_sites" not in st.session_state:
        st.session_state.infection_sites = [""]
    col_site1, col_site2 = st.columns([3, 1])
    with col_site1:
        st.write("**感染部位**")
    with col_site2:
        if st.button("➕ 添加感染部位"):
            st.session_state.infection_sites.append("")
            st.rerun()

    site_options = [
        "请选择", "未查明", "泌尿系统感染", "手术部位感染", "血流感染", "上呼吸道感染", "下呼吸道感染",
        "骨关节感染", "中枢神经系统感染", "心血管系统感染", "消化系统感染", "生殖系统感染",
        "皮肤软组织感染", "全身性感染", "眼/耳/鼻/喉/口腔感染"
    ]
    for i in range(len(st.session_state.infection_sites)):
        col1, col2 = st.columns([3, 1])
        with col1:
            site = st.selectbox(f"感染部位 {i+1}", site_options, key=f"site_{i}")
            st.session_state.infection_sites[i] = site
        with col2:
            if i > 0 and st.button("删除", key=f"del_site_{i}"):
                st.session_state.infection_sites.pop(i)
                st.rerun()
    感染程度 = st.selectbox("感染程度", ["请选择", "严重", "一般", "较轻", "预防"])

    # ---------- 病原菌 + MIC（含目标血药浓度） ----------
    st.markdown("## 三、病原菌 + MIC")
    pathogen_options = ["无", "白色念珠菌", "皮氏不动杆菌", "鲍曼不动杆菌", "鲍曼不动杆菌复合菌",
                        "铜绿假单胞菌", "肺炎克雷伯菌", "大肠埃希菌", "产气克雷伯菌",
                        "奇异变形杆菌", "粪产碱菌粪亚种", "其他"]
    if "target_conc_list" not in st.session_state:
        st.session_state.target_conc_list = []
    if st.button("➕ 添加病原菌+MIC"):
        st.session_state.pathogens.append("")
        st.session_state.mics.append(0)
        st.session_state.target_conc_list.append(0.0)
        st.rerun()

    for i in range(len(st.session_state.pathogens)):
        col1, col2, col3 = st.columns([2, 1.5, 1.5])
        current_val = st.session_state.pathogens[i]
        if current_val not in pathogen_options:
            current_val = "其他" if current_val else "无"
        with col1:
            p = st.selectbox(f"病原菌 {i+1} *", pathogen_options,
                             index=pathogen_options.index(current_val) if current_val in pathogen_options else 0,
                             key=f"p_{i}")
        with col2:
            mic_options = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
            m = st.selectbox(f"MIC {i+1} (mg/L) *", mic_options,
                             index=mic_options.index(st.session_state.mics[i]) if st.session_state.mics[i] in mic_options else 0,
                             key=f"m_{i}",
                             format_func=lambda x: "请选择" if x == 0 else str(x))
        with col3:
            target_conc = st.number_input(f"目标血药浓度 {i+1} (mg/L)", min_value=0.0,
                                         value=st.session_state.target_conc_list[i] if i < len(st.session_state.target_conc_list) else 0.0,
                                         step=0.1, key=f"tc_{i}")
            if i < len(st.session_state.target_conc_list):
                st.session_state.target_conc_list[i] = target_conc
            else:
                st.session_state.target_conc_list.append(target_conc)
        st.session_state.pathogens[i] = p
        st.session_state.mics[i] = m

    # ---------- 肾功能 ----------
    st.markdown("## 四、肾功能")
    col1, col2, col3 = st.columns(3)
    with col1:
        肌酐 = st.text_input("肌酐 (Cr, μmol/L)", placeholder="无/数值")
    with col2:
        eGFR = st.text_input("eGFR (mL/min/1.73m²)", placeholder="无/数值")
    with col3:
        尿酸 = st.text_input("尿酸 (UA, μmol/L)", placeholder="无/数值")
    尿素 = st.text_input("尿素 (Urea, mmol/L)", placeholder="无/数值")

    # ---------- 肝功能（折叠） ----------
    with st.expander("**五、肝功能**", expanded=False):
        col1, col2, col3 = st.columns(3)
        总胆红素 = col1.text_input("总胆红素 (TBIL, μmol/L)", placeholder="无/数值")
        直接胆红素 = col2.text_input("直接胆红素 (DBIL, μmol/L)", placeholder="无/数值")
        间接胆红素 = col3.text_input("间接胆红素 (IBIL, μmol/L)", placeholder="无/数值")
        col4, col5, col6 = st.columns(3)
        总蛋白 = col4.text_input("总蛋白 (TP, g/L)", placeholder="无/数值")
        白蛋白 = col5.text_input("白蛋白 (ALB, g/L)", placeholder="无/数值")
        球蛋白 = col6.text_input("球蛋白 (GLB, g/L)", placeholder="无/数值")
        col7, col8, col9 = st.columns(3)
        ALT = col7.text_input("ALT (U/L)", placeholder="无/数值")
        AST = col8.text_input("AST (U/L)", placeholder="无/数值")
        AST_ALT = col9.text_input("AST:ALT", placeholder="无/数值")
        col10, col11 = st.columns(2)
        碱性磷酸酶 = col10.text_input("ALP (U/L)", placeholder="无/数值")
        谷氨酰氨基转移酶 = col11.text_input("GGT (U/L)", placeholder="无/数值")

    # ---------- 血常规 & 感染指标（折叠） ----------
    with st.expander("**六、血常规 & 感染指标**", expanded=False):
        col1, col2, col3 = st.columns(3)
        WBC = col1.text_input("白细胞 (WBC, ×10⁹/L)", placeholder="无/数值")
        中性粒细胞 = col2.text_input("中性粒细胞百分比 (%)", placeholder="无/数值")
        血小板计数 = col3.text_input("血小板计数 (PLT, ×10⁹/L)", placeholder="无/数值")
        col4, col5, col6 = st.columns(3)
        CRP = col4.text_input("C反应蛋白 (CRP, mg/L)", placeholder="无/数值")
        SAA = col5.text_input("血清淀粉样蛋白A (SAA, mg/L)", placeholder="无/数值")
        PCT = col6.text_input("降钙素原 (PCT, ng/mL)", placeholder="无/数值")

    # ---------- 血药浓度监测（折叠） ----------
    with st.expander("**七、血药浓度监测**", expanded=False):
        col1, col2 = st.columns(2)
        c3h = col1.text_input("第5剂前3h浓度（mg/L）", placeholder="无/数值")
        c05h = col2.text_input("第5剂前0.5h浓度（mg/L）", placeholder="无/数值")

    # 自定义给药方案（独立，不折叠）
    st.markdown("### 八、自定义给药方案（用于计算达标概率）")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        custom_dose = st.number_input("剂量 (g)", min_value=0.25, value=1.0, step=0.25, key="custom_dose")
    with col_d2:
        custom_infusion = st.number_input("输注Time (h)", min_value=0.5, value=0.5, step=0.5, key="custom_infusion")
    with col_d3:
        custom_interval = st.number_input("间隔Time (h)", min_value=6, value=8, step=2, key="custom_interval")

if st.button("📊 计算自定义方案达标概率"):
    # ---------- 感染部位检查 ----------
    sites = st.session_state.infection_sites
    if isinstance(sites, list):
        valid_sites = [s for s in sites if s not in ["请选择", ""]]
        if not valid_sites:
            st.warning("请选择至少一个感染部位")
            st.stop()
    else:
        if sites in ["请选择", "", None]:
            st.warning("请选择感染部位")
            st.stop()
        valid_sites = [sites]

    if 感染程度 == "请选择":
        st.warning("请选择感染程度")
        st.stop()

    try:
        wt = float(体重) if str(体重).strip() not in ["无", ""] else 65.0

        dose_custom = custom_dose
        inf_time = custom_infusion
        interval_custom = custom_interval

        age_val = None
        if 年龄 and str(年龄).strip() not in ["无", ""]:
            try: age_val = float(年龄)
            except: pass
        scr_val = None
        if 肌酐 and str(肌酐).strip() not in ["无", ""]:
            try: scr_val = float(肌酐)
            except: pass
        性别_str = 性别 if 性别 in ["男", "女"] else "男"

        crcl_val = None
        if scr_val is not None and age_val is not None and wt > 0:
            crcl_val = calc_crcl(scr_val, age_val, wt, 性别_str)
        egfr_val = None
        if eGFR and str(eGFR).strip() not in ["无", ""]:
            try: egfr_val = float(eGFR)
            except: pass
        if egfr_val is None and scr_val is not None and age_val is not None and 性别_str in ["男","女"]:
            egfr_val = calc_mdrd(scr_val, age_val, 性别_str)
        if egfr_val is None and crcl_val is not None:
            egfr_val = crcl_val
        if egfr_val is None:
            egfr_val = 90.0

        # 分层 CL/Vd
        if egfr_val >= 130:
            base_cl_60 = 10.0; vd_custom = 0.32
        elif egfr_val >= 90:
            base_cl_60 = 8.0; vd_custom = 0.28
        elif egfr_val >= 60:
            base_cl_60 = 7.0; vd_custom = 0.28
        elif egfr_val >= 45:
            base_cl_60 = 6.5; vd_custom = 0.26
        elif egfr_val >= 30:
            base_cl_60 = 5.5; vd_custom = 0.26
        elif egfr_val >= 15:
            base_cl_60 = 4.5; vd_custom = 0.24
        else:
            base_cl_60 = 4.0; vd_custom = 0.24

        weight_based_cl = base_cl_60 + 0.35 * ((wt - 60.0) / 5.0)
        weight_based_cl = max(4.0, weight_based_cl)
        age_factor = 1.0 - 0.005 * (age_val - 45) if age_val else 1.0
        age_factor = max(0.5, min(1.2, age_factor))
        severe_factor_custom = 1.3 if is_severe(体温, WBC) else 1.0
        cl_custom = weight_based_cl * age_factor * severe_factor_custom

        # 目标浓度 / MIC
        raw_mic = st.session_state.mics[0] if st.session_state.mics else 0
        if raw_mic == 0:
            raw_mic = 2.0
        user_target_conc = st.session_state.target_conc_list[0] if st.session_state.target_conc_list else 0.0

        if user_target_conc > 0:
            target_conc = user_target_conc
            target_mic = raw_mic
            use_direct_conc = True
        else:
            severity = 感染程度 if 感染程度 not in ["请选择", ""] else "严重"
            if severity == "严重":
                target_mic = 4.0 * raw_mic
            else:
                target_mic = 1.0 * raw_mic
            target_conc = target_mic
            use_direct_conc = False

        # 模拟
        t_full, conc_full = simulate_conc_full(dose_custom, interval_custom, cl_custom, vd_custom, wt, inf_time)

        # 波动度
        ke = cl_custom / (vd_custom * wt)
        t_half = np.log(2) / ke if ke > 0 else 2.0
        steady_start = 5 * t_half
        mask = t_full >= steady_start
        fluctuation = 0.0
        if np.any(mask):
            conc_steady = conc_full[mask]
            dt = t_full[1] - t_full[0]
            cycle_len = int(interval_custom / dt) if dt > 0 else 1
            if len(conc_steady) > 3 * cycle_len:
                recent = conc_steady[-3 * cycle_len:]
                peak = float(np.max(recent))
                trough = float(np.min(recent))
                fluctuation = peak / trough if trough > 0 else 0.0
            else:
                peak = float(np.max(conc_steady))
                trough = float(np.min(conc_steady))
                fluctuation = peak / trough if trough > 0 else 0.0
        else:
            trough = 0.0

        # 达标评价
        if use_direct_conc:
            if trough >= target_conc:
                target_attainment = f"Yes (trough={trough:.1f} ≥ target={target_conc})"
            else:
                target_attainment = f"No (trough={trough:.1f} < target={target_conc})"
        else:
            ft_custom = calculate_ft_mic(conc_full, [target_mic])[0]
            target_attainment = f"{ft_custom:.1f}% (target MIC={target_mic})"

        # 绘图
        st.subheader("📉 自定义方案血药浓度曲线（第5次给药稳态）")
        fig, ax = plt.subplots(figsize=(10, 5))
        start_cycle = 4 * interval_custom
        end_cycle = 5 * interval_custom
        if end_cycle > t_full[-1]:
            end_cycle = t_full[-1]
            start_cycle = max(0, end_cycle - interval_custom)
        idx_start = np.searchsorted(t_full, start_cycle)
        idx_end = np.searchsorted(t_full, end_cycle)
        t_cycle = t_full[idx_start:idx_end] - start_cycle
        conc_cycle = conc_full[idx_start:idx_end]

        ax.plot(t_cycle, conc_cycle, 'b-', lw=2, label=f'{dose_custom}g q{interval_custom}h inf{inf_time}h')
        if use_direct_conc:
            ax.axhline(target_conc, ls='--', color='red', lw=2, label=f'Target Conc.={target_conc} mg/L')
        else:
            ax.axhline(target_mic, ls='--', color='red', lw=2, label=f'Target MIC={target_mic}')
        ax.axhline(raw_mic, ls=':', color='gray', alpha=0.5, label=f'Raw MIC={raw_mic}')

        if len(conc_cycle) > 0:
            peak_idx = np.argmax(conc_cycle)
            peak_conc = conc_cycle[peak_idx]
            peak_time = t_cycle[peak_idx]
            trough_conc = conc_cycle[-1]
            trough_time = t_cycle[-1]
            ax.scatter(peak_time, peak_conc, color='green', s=80, zorder=5, label=f'Peak {peak_conc:.1f}')
            ax.scatter(trough_time, trough_conc, color='purple', s=60, zorder=5, label=f'Trough {trough_conc:.1f}')
            n = 1
            while True:
                t_mark = peak_time + n * t_half
                if t_mark > t_cycle[-1]:
                    break
                idx = np.argmin(np.abs(t_cycle - t_mark))
                conc_mark = conc_cycle[idx]
                ax.scatter(t_mark, conc_mark, color='orange', s=60, zorder=5,
                           label=f'{n} Half-life ({n*t_half:.1f}h)' if n == 1 else None)
                ax.annotate(f'{conc_mark:.1f}', xy=(t_mark, conc_mark), xytext=(5,5),
                            textcoords='offset points', fontsize=8, color='orange')
                ax.axvline(t_mark, ls=':', color='orange', alpha=0.5)
                n += 1
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (mg/L)')
        ax.set_title('Meropenem Custom Regimen Steady-State')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("剂量", f"{dose_custom} g")
            st.metric("间隔", f"q{interval_custom}h")
        with col2:
            st.metric("输注时间", f"{inf_time}小时")
            st.metric("波动度 (峰/谷)", f"{fluctuation:.2f}")
            st.metric("理论半衰期", f"{t_half:.1f} h")

        ft_custom_all = calculate_ft_mic(conc_full, [1,2,4,8,16])
        ft_df = pd.DataFrame({"MIC (mg/L)": ["1","2","4","8","16"], "%T>MIC": [f"{v:.1f}%" for v in ft_custom_all]})
        st.dataframe(ft_df, hide_index=True, width='stretch')

    except Exception as e:
        st.error(f"计算失败：{str(e)}")

# ----------------------------- 生成个体化给药方案（修复版） -----------------------------
if st.button("✅ 生成个体化给药方案"):
    # ---------- 感染部位有效性检查 ----------
    sites = st.session_state.infection_sites
    if isinstance(sites, list):
        valid_sites = [s for s in sites if s not in ["请选择", ""]]
        if not valid_sites:
            st.warning("请选择至少一个感染部位")
            st.stop()
    else:
        if sites in ["请选择", "", None]:
            st.warning("请选择感染部位")
            st.stop()
        valid_sites = [sites]

    if 感染程度 == "请选择":
        st.warning("请选择感染程度")
        st.stop()

    try:
        # ==================== 患者参数解析 ====================
        try:
            wt = float(体重) if str(体重).strip() not in ["无", ""] else 65.0
        except:
            wt = 65.0

        age_val = None
        if 年龄 and str(年龄).strip() not in ["无", ""]:
            try: age_val = float(年龄)
            except: pass
        scr_val = None
        if 肌酐 and str(肌酐).strip() not in ["无", ""]:
            try: scr_val = float(肌酐)
            except: pass
        性别_str = 性别 if 性别 in ["男", "女"] else "男"

        crcl_val = None
        if scr_val is not None and age_val is not None and wt > 0:
            crcl_val = calc_crcl(scr_val, age_val, wt, 性别_str)
        egfr_mdrd = None
        if scr_val is not None and age_val is not None:
            egfr_mdrd = calc_mdrd(scr_val, age_val, 性别_str)

        重症 = is_severe(体温, WBC)
        耐药 = is_resistant(st.session_state.mics)
        has_cns = "中枢神经系统感染" in valid_sites if isinstance(valid_sites, list) else (valid_sites == "中枢神经系统感染")

        # ==================== 初始经验推荐 ====================
        if 重症 or 耐药 or has_cns:
            base_dose = 2.0; base_interval = 8
        else:
            base_dose = 1.0; base_interval = 8
        infection_dose = base_dose

        # ==================== 肾功能调整（基于 eGFR 分级） ====================
        egfr_val = None
        if eGFR and str(eGFR).strip() not in ["无", ""]:
            try: egfr_val = float(eGFR)
            except: pass
        if egfr_val is None and scr_val is not None and age_val is not None and 性别_str in ["男","女"]:
            egfr_val = calc_mdrd(scr_val, age_val, 性别_str)
        if egfr_val is None and crcl_val is not None:
            egfr_val = crcl_val
        if egfr_val is None:
            egfr_val = 90.0

        if egfr_val >= 130:
            base_dose = 2.0; base_interval = 8
        elif 90 <= egfr_val < 130:
            pass
        elif 60 <= egfr_val < 90:
            base_dose = 1.0; base_interval = 12
        elif 45 <= egfr_val < 60:
            base_dose = 1.0; base_interval = 12
        elif 30 <= egfr_val < 45:
            base_dose = 0.5; base_interval = 8
        elif 15 <= egfr_val < 30:
            base_dose = 0.5; base_interval = 12
        else:
            base_dose = 0.5; base_interval = 12

        剂量 = max(0.25, min(2.0, round(base_dose * 4) / 4))
        间隔 = base_interval
        输注_time = 0.5
        renal_dose = 剂量
        renal_interval = 间隔

        # ==================== 分层 PPK 参数 ====================
        if egfr_val >= 130:
            base_cl_60 = 10.0; vd_ppk = 0.32
        elif egfr_val >= 90:
            base_cl_60 = 8.0; vd_ppk = 0.28
        elif egfr_val >= 60:
            base_cl_60 = 7.0; vd_ppk = 0.28
        elif egfr_val >= 45:
            base_cl_60 = 6.5; vd_ppk = 0.26
        elif egfr_val >= 30:
            base_cl_60 = 5.5; vd_ppk = 0.26
        elif egfr_val >= 15:
            base_cl_60 = 4.5; vd_ppk = 0.24
        else:
            base_cl_60 = 4.0; vd_ppk = 0.24

        weight_based_cl = base_cl_60 + 0.35 * ((wt - 60.0) / 5.0)
        weight_based_cl = max(4.0, weight_based_cl)

        age_factor = 1.0 - 0.005 * (age_val - 45) if age_val else 1.0
        age_factor = max(0.5, min(1.2, age_factor))
        severe_factor = 1.3 if 重症 else 1.0

        cl_ppk = weight_based_cl * age_factor * severe_factor
        cl_final = cl_ppk
        vd_final = vd_ppk

        # ==================== 目标 MIC 与目标血药浓度 ====================
        raw_mic = st.session_state.mics[0] if st.session_state.mics else 0
        if raw_mic == 0:
            raw_mic = 2.0   # 未选择默认 2 mg/L
        user_target_conc = st.session_state.target_conc_list[0] if st.session_state.target_conc_list else 0.0

        severity = 感染程度 if 感染程度 not in ["请选择", ""] else "严重"

        if user_target_conc > 0:
            target_conc = user_target_conc
            target_mic = raw_mic
            use_direct_conc = True
        else:
            if severity == "严重":
                target_mic = 4.0 * raw_mic
            else:
                target_mic = 1.0 * raw_mic
            target_conc = target_mic
            use_direct_conc = False

        mic_values = [1, 2, 4, 8, 16]

        # 初始化 best_ft
        best_ft = 0.0

        # 处理 TDM 实测值
        obs_c = None
        if c05h and str(c05h).strip() not in ["无", ""]:
            try: obs_c = float(c05h)
            except: pass

        ml_adjusted = False
        if obs_c is None:
            ml_model = st.session_state.get("pk_model", None)
            ml_features = st.session_state.get("pk_features", None)

            if ml_model is not None and ml_features is not None:
                try:
                    tmp_record = {
                        "编号": 编号, "日期": str(日期),
                        "年龄": 年龄, "性别": 性别, "身高": 身高, "体重": 体重,
                        "体温": 体温, "心率": 心率, "血压": 血压,
                        "肌酐": 肌酐, "eGFR": eGFR, "尿酸": 尿酸, "尿素": 尿素,
                        "总胆红素": 总胆红素, "直接胆红素": 直接胆红素, "间接胆红素": 间接胆红素,
                        "总蛋白": 总蛋白, "白蛋白": 白蛋白, "球蛋白": 球蛋白,
                        "丙氨酸氨基转移酶ALT": ALT, "AST": AST, "AST:ALT": AST_ALT,
                        "碱性磷酸酶": 碱性磷酸酶, "谷氨酰氨基转移酶": 谷氨酰氨基转移酶,
                        "白细胞(WBC)": WBC, "中性粒细胞": 中性粒细胞, "血小板计数": 血小板计数,
                        "CRP": CRP, "血清淀粉样C蛋白SAA": SAA, "PCT": PCT,
                        "感染部位": valid_sites, "感染程度": 感染程度,
                        "病原菌列表": st.session_state.pathogens, "MIC列表": st.session_state.mics,
                        "剂量": 剂量, "间隔": 间隔, "输注时间": 输注_time,
                        "滴注速度": infusion_rate if 'infusion_rate' in dir() else 0,
                        "给药前3h浓度": c3h, "给药前0.5h浓度": None, "结论类型": None
                    }
                    df_tmp = pd.DataFrame([tmp_record])
                    df_tmp_eng = add_engineered_features(df_tmp)

                    for feat in ml_features:
                        if feat not in df_tmp_eng.columns:
                            df_tmp_eng[feat] = 0

                    cat_cols = ["性别", "感染部位_首位", "感染程度", "病原菌列表"]
                    for col in cat_cols:
                        if col in df_tmp_eng.columns:
                            le = LabelEncoder()
                            df_tmp_eng[col] = df_tmp_eng[col].astype(str)
                            df_tmp_eng[col] = le.fit_transform(df_tmp_eng[col])

                    X_pred = df_tmp_eng[ml_features].apply(pd.to_numeric, errors='coerce').fillna(0)
                    pred_ml = ml_model.predict(X_pred)[0]

                    # 根据达标模式进行双向调整
                    if use_direct_conc:
                        t_curr, conc_curr = simulate_conc_full(剂量, 间隔, cl_final, vd_final, wt, 输注_time)
                        ke_curr = cl_final / (vd_final * wt)
                        t_half_curr = np.log(2)/ke_curr if ke_curr>0 else 2.0
                        steady_curr = 5 * t_half_curr
                        mask_curr = t_curr >= steady_curr
                        trough_curr = conc_curr[mask_curr][-1] if np.any(mask_curr) else 0.0
                        if trough_curr < target_conc:
                            ml_adjusted = True
                            found = False
                            infusion_options = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                            interval_options = sorted(set([间隔, 6]))
                            dose_step, max_dose = 0.25, 2.0
                            for inf in infusion_options:
                                if found: break
                                for intv in interval_options:
                                    if found: break
                                    dose_try = 剂量
                                    while dose_try <= max_dose:
                                        t_full, conc_full = simulate_conc_full(dose_try, intv, cl_final, vd_final, wt, inf)
                                        ke_temp = cl_final / (vd_final * wt)
                                        t_half_temp = np.log(2)/ke_temp if ke_temp>0 else 2.0
                                        steady = 5 * t_half_temp
                                        mask = t_full >= steady
                                        if np.any(mask):
                                            trough = conc_full[mask][-1]
                                            if trough >= target_conc:
                                                剂量, 间隔, 输注_time = dose_try, intv, inf
                                                best_t_full, best_conc_full = t_full, conc_full
                                                found = True
                                                break
                                        dose_try += dose_step
                                        dose_try = round(dose_try * 4) / 4
                            if not found:
                                st.warning("已使用最大强度方案，仍未达到目标谷浓度。")
                        elif trough_curr > target_conc * 1.5:
                            ml_adjusted = True
                            dose_opts = sorted(np.arange(0.5, 剂量 + 0.01, 0.25), reverse=False)
                            inf_opts = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                            intv_opts = [6,8,12,24]
                            best_safe = None
                            for d in dose_opts:
                                if best_safe: break
                                for inf in inf_opts:
                                    if best_safe: break
                                    for intv in intv_opts:
                                        t_full, conc_full = simulate_conc_full(d, intv, cl_final, vd_final, wt, inf)
                                        ke_temp = cl_final / (vd_final * wt)
                                        t_half_temp = np.log(2)/ke_temp if ke_temp>0 else 2.0
                                        steady = 5 * t_half_temp
                                        mask = t_full >= steady
                                        if np.any(mask):
                                            trough = conc_full[mask][-1]
                                            if trough >= target_conc and trough <= target_conc * 2.5:
                                                best_safe = (d, intv, inf, t_full, conc_full, trough)
                                                break
                            if best_safe:
                                剂量, 间隔, 输注_time = best_safe[0], best_safe[1], best_safe[2]
                                best_t_full, best_conc_full = best_safe[3], best_safe[4]
                            else:
                                st.warning("无法找到安全且达标的降级方案，维持原方案。")
                        else:
                            ml_adjusted = True
                            best_t_full, best_conc_full = t_curr, conc_curr
                    else:
                        # 模式2：基于 %T>MIC
                        t_curr, conc_curr = simulate_conc_full(剂量, 间隔, cl_final, vd_final, wt, 输注_time)
                        ft_curr = calculate_ft_mic(conc_curr, [target_mic])[0]
                        if ft_curr < 90:
                            ml_adjusted = True
                            found = False
                            infusion_options = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                            interval_options = sorted(set([间隔, 6]))
                            dose_step, max_dose = 0.25, 2.0
                            for inf in infusion_options:
                                if found: break
                                for intv in interval_options:
                                    if found: break
                                    dose_try = 剂量
                                    while dose_try <= max_dose:
                                        t_full, conc_full = simulate_conc_full(dose_try, intv, cl_final, vd_final, wt, inf)
                                        ft_mic = calculate_ft_mic(conc_full, [target_mic])
                                        current_ft = ft_mic[0]
                                        if current_ft >= 90:
                                            剂量, 间隔, 输注_time = dose_try, intv, inf
                                            best_ft = current_ft
                                            best_t_full, best_conc_full = t_full, conc_full
                                            found = True
                                            break
                                        dose_try += dose_step
                                        dose_try = round(dose_try * 4) / 4
                            if not found:
                                st.warning("已使用最大强度方案，仍不达标。")
                        elif ft_curr >= 90 and pred_ml > target_mic * 2:
                            ml_adjusted = True
                            dose_opts = sorted(np.arange(0.5, 剂量 + 0.01, 0.25), reverse=False)
                            inf_opts = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                            intv_opts = [6,8,12,24]
                            best_safe = None
                            for d in dose_opts:
                                if best_safe: break
                                for inf in inf_opts:
                                    if best_safe: break
                                    for intv in intv_opts:
                                        t_full, conc_full = simulate_conc_full(d, intv, cl_final, vd_final, wt, inf)
                                        ft_mic = calculate_ft_mic(conc_full, [target_mic])
                                        current_ft = ft_mic[0]
                                        if current_ft >= 90:
                                            ke_temp = cl_final / (vd_final * wt)
                                            t_half_temp = np.log(2)/ke_temp if ke_temp>0 else 2.0
                                            steady = 5 * t_half_temp
                                            mask = t_full >= steady
                                            if np.any(mask):
                                                trough = conc_full[mask][-1]
                                                if trough <= target_mic * 3:
                                                    best_safe = (d, intv, inf, current_ft, t_full, conc_full, trough)
                                                    break
                            if best_safe:
                                剂量, 间隔, 输注_time = best_safe[0], best_safe[1], best_safe[2]
                                best_ft = best_safe[3]
                                best_t_full, best_conc_full = best_safe[4], best_safe[5]
                            else:
                                st.warning("无法找到安全达标方案，维持原方案。")
                        else:
                            ml_adjusted = True
                            cl_est, vd_est = bayesian_estimate(0.5, pred_ml, 剂量, 间隔, wt, cl_ppk, vd_ppk)
                            cl_final, vd_final = cl_est, vd_est
                            t_full, conc_full = simulate_conc_full(剂量, 间隔, cl_final, vd_final, wt, 输注_time)
                            ft_mic = calculate_ft_mic(conc_full, [target_mic])
                            best_ft = ft_mic[0]
                            best_t_full, best_conc_full = t_full, conc_full
                except Exception as e:
                    st.warning(f"ML prediction failed ({e}), using PPK-only regimen.")
            else:
                st.warning("⚠️ 未训练PK机器学习模型，使用纯PPK方案。")

        if not ml_adjusted:
            bayes_adjusted = False
            if obs_c is not None and obs_c > 0:
                cl_est, vd_est = bayesian_estimate(0.5, obs_c, 剂量, 间隔, wt, cl_ppk, vd_ppk)
                cl_final, vd_final = cl_est, vd_est
                target_auc = 400
                auc_current = (剂量 * 1000) / cl_est if cl_est > 0 else 400
                if auc_current > 0:
                    剂量 = 剂量 * (target_auc / auc_current)
                剂量 = max(0.25, min(2.0, round(剂量 * 4) / 4))
                bayes_adjusted = True

            total_hours = max(48, 间隔 * 10, 5 * (np.log(2) / (cl_final / (vd_final * wt)) if cl_final > 0 else 2.0))
            t_init, conc_init = simulate_conc_full(剂量, 间隔, cl_final, vd_final, wt, 输注_time)

            if use_direct_conc:
                ke_init = cl_final / (vd_final * wt)
                t_half_init = np.log(2)/ke_init if ke_init>0 else 2.0
                steady_init = 5 * t_half_init
                mask_init = t_init >= steady_init
                trough_init = conc_init[mask_init][-1] if np.any(mask_init) else 0.0
                if trough_init < target_conc:
                    adjusted = True
                    found = False
                    infusion_options = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                    interval_options = sorted(set([间隔, 6]))
                    dose_step, max_dose = 0.25, 2.0
                    for inf in infusion_options:
                        if found: break
                        for intv in interval_options:
                            if found: break
                            dose_try = 剂量
                            while dose_try <= max_dose:
                                t_full, conc_full = simulate_conc_full(dose_try, intv, cl_final, vd_final, wt, inf)
                                ke_temp = cl_final / (vd_final * wt)
                                t_half_temp = np.log(2)/ke_temp if ke_temp>0 else 2.0
                                steady = 5 * t_half_temp
                                mask = t_full >= steady
                                if np.any(mask):
                                    trough = conc_full[mask][-1]
                                    if trough >= target_conc:
                                        final_dose, final_interval, final_inf = dose_try, intv, inf
                                        best_t_full, best_conc_full = t_full, conc_full
                                        found = True
                                        break
                                dose_try += dose_step
                                dose_try = round(dose_try * 4) / 4
                    if not found:
                        final_dose, final_interval, final_inf = 2.0, 6, 5.0
                        t_full, conc_full = simulate_conc_full(final_dose, final_interval, cl_final, vd_final, wt, final_inf)
                        best_t_full, best_conc_full = t_full, conc_full
                else:
                    best_t_full, best_conc_full = t_init, conc_init
                    final_dose, final_interval, final_inf = 剂量, 间隔, 输注_time
            else:
                ft_init = calculate_ft_mic(conc_init, [target_mic])[0]
                best_ft = ft_init
                best_t_full, best_conc_full = t_init, conc_init
                final_dose, final_interval, final_inf = 剂量, 间隔, 输注_time
                adjusted = False

                if ft_init < 90:
                    adjusted = True
                    found = False
                    infusion_options = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                    interval_options = sorted(set([间隔, 6]))
                    dose_step, max_dose = 0.25, 2.0
                    for inf in infusion_options:
                        if found: break
                        for intv in interval_options:
                            if found: break
                            dose_try = 剂量
                            while dose_try <= max_dose:
                                t_full, conc_full = simulate_conc_full(dose_try, intv, cl_final, vd_final, wt, inf)
                                ft_mic = calculate_ft_mic(conc_full, [target_mic])
                                current_ft = ft_mic[0]
                                if current_ft >= 90:
                                    final_dose, final_interval, final_inf = dose_try, intv, inf
                                    best_ft = current_ft
                                    best_t_full, best_conc_full = t_full, conc_full
                                    found = True
                                    break
                                dose_try += dose_step
                                dose_try = round(dose_try * 4) / 4
                    if not found:
                        final_dose, final_interval, final_inf = 2.0, 6, 5.0
                        t_full, conc_full = simulate_conc_full(final_dose, final_interval, cl_final, vd_final, wt, final_inf)
                        ft_mic = calculate_ft_mic(conc_full, [target_mic])
                        best_ft = ft_mic[0]
                        best_t_full, best_conc_full = t_full, conc_full

                if not adjusted and not bayes_adjusted:
                    best_ft = ft_init
        else:
            final_dose, final_interval, final_inf = 剂量, 间隔, 输注_time
            if 'best_t_full' not in locals():
                t_full, conc_full = simulate_conc_full(final_dose, final_interval, cl_final, vd_final, wt, final_inf)
                best_t_full, best_conc_full = t_full, conc_full

        # ==================== 最终模拟数据 ====================
        t_full, conc_full = best_t_full, best_conc_full

        # 波动度计算
        ke = cl_final / (vd_final * wt)
        t_half = np.log(2) / ke if ke > 0 else 2.0
        steady_start = 5 * t_half
        mask_steady = t_full >= steady_start
        pred_peak = pred_trough = 0.0
        fluctuation = 0.0
        if np.any(mask_steady):
            conc_steady = conc_full[mask_steady]
            dt = t_full[1] - t_full[0]
            cycle_len = int(final_interval / dt) if dt > 0 else 1
            if len(conc_steady) > 3 * cycle_len:
                recent = conc_steady[-3 * cycle_len:]
                pred_peak = float(np.max(recent))
                pred_trough = float(np.min(recent))
                fluctuation = pred_peak / pred_trough if pred_trough > 0 else 0.0
            else:
                pred_peak = float(np.max(conc_steady))
                pred_trough = float(np.min(conc_steady))
                fluctuation = pred_peak / pred_trough if pred_trough > 0 else 0.0
        pred_auc24 = (final_dose * 1000) / cl_final if cl_final > 0 else 0.0

        trough_conc = pred_trough if pred_trough > 0 else conc_full[-1]

        # 结论判断
        c05h_num = -1
        if c05h and str(c05h).strip() not in ["无", ""]:
            try: c05h_num = float(c05h)
            except: pass
        if obs_c is not None and c05h_num > 30:
            conclusion = "血药浓度过高"
        elif use_direct_conc:
            if trough_conc >= target_conc:
                conclusion = "血药浓度达标"
            else:
                conclusion = "血药浓度过低"
        else:
            if best_ft >= 90:
                conclusion = "血药浓度达标"
            else:
                conclusion = "血药浓度过低"

        # ==================== 规范化结果解释 ====================
        site_str = valid_sites[0] if valid_sites else "未查明"
        severity_str = severity
        if raw_mic <= 1:
            sensitivity = "敏感"
        elif 2 <= raw_mic <= 4:
            sensitivity = "中介"
        else:
            sensitivity = "耐药"
        pathogen_str = st.session_state.pathogens[0] if st.session_state.pathogens and st.session_state.pathogens[0] else "未查明病原菌"
        if pathogen_str == "无" or not pathogen_str:
            pathogen_str = "未查明病原菌"
        if egfr_val >= 130:
            renal_str = "亢进"
        elif 90 <= egfr_val < 130:
            renal_str = "正常"
        elif 60 <= egfr_val < 90:
            renal_str = "轻度下降"
        elif 45 <= egfr_val < 60:
            renal_str = "轻至中度下降"
        elif 30 <= egfr_val < 45:
            renal_str = "中至重度下降"
        elif 15 <= egfr_val < 30:
            renal_str = "重度下降"
        else:
            renal_str = "衰竭"
        if use_direct_conc:
            target_conc_value = target_conc
            mic_multiplier = "（手动输入）"
        else:
            target_conc_value = target_mic
            mic_multiplier = f"（{4 if severity_str == '严重' else 1}倍MIC值）"
        if conclusion == "血药浓度达标":
            status = "已达标"
            suggestion = "可应用该方案并进行随访。"
        else:
            status = "未达标"
            suggestion = "应调整剂量。"
        explanation = f"患者为{site_str}{severity_str}感染，{sensitivity}的{pathogen_str}，且肾功能{renal_str}，目标血药浓度为{target_conc_value}ug/ml{mic_multiplier}，{status}，{suggestion}"

        # ==================== 保存记录 ====================
        record = {
            "编号": 编号, "日期": str(日期),
            "年龄": 年龄, "性别": 性别, "身高": 身高, "体重": 体重, "体温": 体温, "心率": 心率, "血压": 血压,
            "肌酐": 肌酐, "eGFR": eGFR, "尿酸": 尿酸, "尿素": 尿素,
            "总胆红素": 总胆红素, "直接胆红素": 直接胆红素, "间接胆红素": 间接胆红素,
            "总蛋白": 总蛋白, "白蛋白": 白蛋白, "球蛋白": 球蛋白,
            "丙氨酸氨基转移酶ALT": ALT, "AST": AST, "AST:ALT": AST_ALT,
            "碱性磷酸酶": 碱性磷酸酶, "谷氨酰氨基转移酶": 谷氨酰氨基转移酶,
            "白细胞(WBC)": WBC, "中性粒细胞": 中性粒细胞, "血小板计数": 血小板计数,
            "CRP": CRP, "血清淀粉样C蛋白SAA": SAA, "PCT": PCT,
            "给药前3h浓度": c3h, "给药前0.5h浓度": c05h,
            "感染部位": valid_sites, "感染程度": 感染程度,
            "病原菌列表": st.session_state.pathogens, "MIC列表": st.session_state.mics,
            "剂量": final_dose, "间隔": final_interval, "输注时间": final_inf,
            "结论类型": conclusion,
            "预测峰浓度": round(pred_peak, 2), "预测谷浓度": round(trough_conc, 2),
            "预测AUC24": round(pred_auc24, 2),
            "目标MIC": target_mic, "%T>MIC": round(best_ft if not use_direct_conc else 0, 1),
            "建议": explanation,
            "record_id": str(uuid.uuid4())
        }
        st.session_state.patient_db.append(record)
        # 普通用户同步到全局数据库
        if st.session_state.user_role == "user":
            global_db.add_record(record)

        # ==================== 展示报告 ====================
        st.success("✅ 报告生成完成")
        colA, colB = st.columns(2)
        with colA:
            st.subheader("💊 个体化给药方案")
            st.metric("推荐剂量", f"{final_dose} g")
            st.metric("给药间隔", f"q{final_interval}h")
            st.metric("输注时间", f"{final_inf}小时")
            st.metric("波动度 (峰/谷)", f"{fluctuation:.2f}")
            if use_direct_conc:
                st.metric("目标谷浓度", f"{target_conc} mg/L")
            else:
                st.metric("目标 MIC", f"{target_mic} mg/L")
                st.metric("%T>MIC", f"{best_ft:.1f}%")
            st.metric("重症", "是" if 重症 else "否")
            st.metric("耐药", "是" if 耐药 else "否")
        with colB:
            st.subheader("📉 血药浓度曲线（第5次给药稳态周期）")
            fig, ax = plt.subplots(figsize=(10, 5))
            start_cycle = 4 * final_interval
            end_cycle = 5 * final_interval
            if end_cycle > t_full[-1]:
                end_cycle = t_full[-1]
                start_cycle = max(0, end_cycle - final_interval)
            idx_start = np.searchsorted(t_full, start_cycle)
            idx_end = np.searchsorted(t_full, end_cycle)
            t_cycle = t_full[idx_start:idx_end] - start_cycle
            conc_cycle = conc_full[idx_start:idx_end]
            ax.plot(t_cycle, conc_cycle, 'b-', lw=2, label='Predicted Conc.')
            if use_direct_conc:
                ax.axhline(target_conc, ls='--', color='red', lw=2, label=f'Target Conc.={target_conc} mg/L')
            else:
                ax.axhline(target_mic, ls='--', color='red', lw=2, label=f'Target MIC={target_mic}')
            ax.axhline(raw_mic, ls=':', color='gray', alpha=0.7, label=f'Raw MIC={raw_mic}')
            if len(conc_cycle) > 0:
                peak_idx = np.argmax(conc_cycle)
                peak_conc = conc_cycle[peak_idx]
                peak_time = t_cycle[peak_idx]
                trough_conc = conc_cycle[-1]
                trough_time = t_cycle[-1]
                ax.scatter(peak_time, peak_conc, color='green', s=80, zorder=5, label=f'Peak {peak_conc:.1f}')
                ax.scatter(trough_time, trough_conc, color='purple', s=60, zorder=5, label=f'Trough {trough_conc:.1f}')
                n = 1
                while True:
                    t_mark = peak_time + n * t_half
                    if t_mark > t_cycle[-1]:
                        break
                    idx = np.argmin(np.abs(t_cycle - t_mark))
                    conc_mark = conc_cycle[idx]
                    ax.scatter(t_mark, conc_mark, color='orange', s=60, zorder=5,
                               label=f'{n} Half-life ({n*t_half:.1f}h)' if n == 1 else None)
                    ax.annotate(f'{conc_mark:.1f}', xy=(t_mark, conc_mark), xytext=(5,5),
                                textcoords='offset points', fontsize=8, color='orange')
                    ax.axvline(t_mark, ls=':', color='orange', alpha=0.5)
                    n += 1
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Concentration (mg/L)')
            ax.set_title('Meropenem Steady-State Concentration (5th Dose)')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=8)
            st.pyplot(fig)

        st.subheader("📄 血药浓度分析报告")
        ft_mic_final = calculate_ft_mic(conc_full, mic_values)
        items = [
            "Conc. 3h before dose", "Conc. 0.5h before dose",
            "Target Attainment",
            "%T>MIC (1 mg/L)", "%T>MIC (2 mg/L)", "%T>MIC (4 mg/L)",
            "%T>MIC (8 mg/L)", "%T>MIC (16 mg/L)"
        ]
        if use_direct_conc:
            target_attainment = "Yes" if trough_conc >= target_conc else "No"
        else:
            target_attainment = f"{best_ft:.1f}%"
        res = [
            str(c3h) if c3h and str(c3h).strip() not in ["无", ""] else "N/A",
            f"{trough_conc:.2f} mg/L (simulated)" if obs_c is None else f"{trough_conc:.2f} mg/L",
            target_attainment,
            f"{ft_mic_final[0]:.1f}%", f"{ft_mic_final[1]:.1f}%", f"{ft_mic_final[2]:.1f}%",
            f"{ft_mic_final[3]:.1f}%", f"{ft_mic_final[4]:.1f}%"
        ]
        tbl = "<table style='width:100%; border-collapse: collapse;'>"
        tbl += "<tr style='background-color:#f2f2f2;'><th style='padding:8px; border:1px solid #ddd;'>项目</th><th style='padding:8px; border:1px solid #ddd;'>结果</th></tr>"
        for i, r in zip(items, res):
            tbl += f"<tr><td style='padding:8px; border:1px solid #ddd;'>{i}</td><td style='padding:8px; border:1px solid #ddd;'>{r}</td></tr>"
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        st.subheader("📌 结果解释与建议")
        st.markdown(explanation)

    except Exception as e:
        st.error(f"方案生成失败：{str(e)}")