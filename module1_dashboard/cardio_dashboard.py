# -*- coding: utf-8 -*-
"""
CardioAI - 模块1: 心血管疾病数据可视化看板
==========================================
一个 Streamlit 应用程序，对 data/心血管疾病.xlsx 进行数据清洗、特征工程与
交互式可视化（Plotly Express）。

运行方式:
    streamlit run module1_dashboard/cardio_dashboard.py

依赖:
    pandas, numpy, streamlit, plotly.express
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 数据文件固定在项目根目录的 data/ 下，用脚本所在位置向上推导，
# 这样无论从哪里启动 streamlit 都能正确找到数据文件。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "心血管疾病.xlsx")

# 描述性映射（原始数据集字段均为数值编码）
GENDER_MAP = {1: "女性", 2: "男性"}
CHOL_MAP = {1: "正常", 2: "偏高", 3: "严重偏高"}
GLUC_MAP = {1: "正常", 2: "偏高", 3: "严重偏高"}
CARDIO_MAP = {0: "无疾病", 1: "有疾病"}

# BMI 类别顺序（中国成人标准: 偏瘦 <18.5, 正常 18.5~23.9, 超重 24~27.9, 肥胖 >=28）
BMI_CATEGORY_ORDER = ["偏瘦", "正常", "超重", "肥胖"]

# 图表配色（dataviz 规范: 类别槽位蓝/橙，两个状态固定映射，不随筛选变化）
SERIES_COLORS = {"无疾病": "#2a78d6", "有疾病": "#eb6834"}

PLOTLY_FONT = {"family": "system-ui, -apple-system, 'Segoe UI', sans-serif"}

# ---------------------------------------------------------------------------
# 全局样式与英雄区（与模块2/3同一套设计令牌：冷调临床 + 心电青绿）
# ---------------------------------------------------------------------------
_CSS = """
<style>
    #MainMenu, [data-testid="stMainMenu"] { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.6rem; }

    /* 指标卡 -> 监护仪读数 */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D8E2E6;
        border-left: 4px solid #0E7C86;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(16, 33, 43, .04);
    }
    [data-testid="stMetricLabel"] { color: #5C6F7A; font-size: 13px; font-weight: 600; }
    [data-testid="stMetricValue"] {
        font-family: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
        font-variant-numeric: tabular-nums;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* 侧边栏 */
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D8E2E6; }
    [data-testid="stSidebar"] h2 { font-size: 15px; color: #10212B; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #5C6F7A; }

    /* 图表区块标题 */
    [data-testid="stHeader"] { background: transparent; }
</style>
"""


def _hero_html(data_path: str, n_raw: int, n_clean: int) -> str:
    """英雄区 HTML：心电图纸网格背景 + 自绘 ECG 波形 + 标题。"""
    return f"""
<div style="background-image:repeating-linear-gradient(90deg,rgba(14,124,134,.05) 0 1px,transparent 1px 26px),repeating-linear-gradient(0deg,rgba(14,124,134,.05) 0 1px,transparent 1px 26px);border-bottom:1px solid #D8E2E6;padding:26px 0 20px;margin-bottom:6px;">
    <svg viewBox="0 0 240 56" preserveAspectRatio="none" style="width:100%;height:52px;display:block;margin-bottom:16px;">
        <path d="M0 44 C 8 44 14 34 22 34 C 30 34 34 44 42 44 L 66 44 C 72 44 75 47 79 47 C 82 47 83 41 86 41 L 89 8 L 97 52 C 102 52 106 44 116 44 L 120 44 C 130 44 138 29 148 31 C 157 33 163 44 174 44 L 240 44"
              fill="none" stroke="#0E7C86" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
              pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"
              style="animation:cardio-draw 1.7s ease-out .1s forwards;" />
    </svg>
    <p style="font-family:ui-monospace,Consolas,monospace;font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#0A5A62;margin:0 0 8px;">CardioAI · Data Monitor</p>
    <h1 style="font-size:clamp(26px,4.5vw,38px);line-height:1.15;margin:0 0 10px;color:#10212B;">心血管疾病数据看板</h1>
    <p style="color:#5C6F7A;margin:0;font-size:14px;">数据来源 <code>{data_path}</code> ｜ 清洗前 {n_raw:,} 条，清洗与特征工程后 {n_clean:,} 条</p>
</div>
<style>
    @keyframes cardio-draw {{ to {{ stroke-dashoffset: 0; }} }}
    @media (prefers-reduced-motion: reduce) {{ svg path {{ animation: none !important; stroke-dashoffset: 0 !important; }} }}
</style>
"""


# ---------------------------------------------------------------------------
# 数据加载与清洗（@st.cache_data 缓存，交互筛选时避免重复计算）
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """从 Excel 加载心血管疾病数据集（xlsx 用 openpyxl 引擎，无需处理编码）。"""
    return pd.read_excel(path, engine="openpyxl")


@st.cache_data
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗 + 特征工程。

    1. 特征工程: age(天) -> age_years(年, 四舍五入); 计算 BMI。
    2. 异常值: 删除 舒张压(ap_lo) >= 收缩压(ap_hi) 的记录;
       仅保留 收缩压 in [90, 250]、舒张压 in [60, 150]。
    3. 类别转换: cholesterol / gluc 数值 -> 描述字符串; 生成 bmi_category。
    """
    df = df.copy()

    # ---- 特征工程 ----
    df["age_years"] = np.round(df["age"] / 365.0).astype(int)  # 天 -> 年(四舍五入)
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)  # BMI = 体重 / (身高m)^2

    # ---- 异常值处理 ----
    df = df[df["ap_lo"] < df["ap_hi"]]  # 舒张压必须小于收缩压
    df = df[df["ap_hi"].between(90, 250)]  # 收缩压范围 [90, 250]
    df = df[df["ap_lo"].between(60, 150)]  # 舒张压范围 [60, 150]

    # ---- 类别转换 ----
    df["gender"] = df["gender"].map(GENDER_MAP)
    df["cholesterol"] = df["cholesterol"].map(CHOL_MAP)
    df["gluc"] = df["gluc"].map(GLUC_MAP)
    df["cardio_label"] = df["cardio"].map(CARDIO_MAP)  # 供图表图例使用

    def _bmi_category(bmi: float) -> str:
        if bmi < 18.5:
            return "偏瘦"
        if bmi < 24.0:
            return "正常"
        if bmi < 28.0:
            return "超重"
        return "肥胖"

    df["bmi_category"] = df["bmi"].apply(_bmi_category)

    return df


# ---------------------------------------------------------------------------
# Streamlit 界面
# ---------------------------------------------------------------------------
def build_ui(df_raw: pd.DataFrame) -> None:
    """渲染侧边栏筛选器、顶部指标与两张 Plotly 图表。"""
    df = clean_and_engineer(df_raw)

    # ---- 侧边栏筛选 ----
    st.sidebar.header("筛选条件")

    age_min, age_max = int(df["age_years"].min()), int(df["age_years"].max())
    age_range = st.sidebar.slider(
        "年龄范围（岁）",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )

    gender_options = sorted(df["gender"].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect("性别", gender_options, default=gender_options)

    cardio_options = sorted(df["cardio_label"].dropna().unique().tolist())
    selected_cardio = st.sidebar.multiselect("心血管疾病", cardio_options, default=cardio_options)

    # 应用筛选
    mask = (
        df["age_years"].between(age_range[0], age_range[1])
        & df["gender"].isin(selected_genders)
        & df["cardio_label"].isin(selected_cardio)
    )
    filtered = df[mask]

    # ---- 全局样式 + 英雄区（ECG 签名）----
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_hero_html(DATA_PATH, len(df_raw), len(df)), unsafe_allow_html=True)

    risk_rate = float(filtered["cardio"].mean()) * 100 if len(filtered) else 0.0
    c1, c2 = st.columns(2)
    c1.metric("筛选后记录数", f"{len(filtered):,}")
    c2.metric("心血管疾病总风险率", f"{risk_rate:.2f}%")

    if filtered.empty:
        st.warning("当前筛选条件下没有记录，请调整侧边栏筛选条件。")
        return

    # ---- 图表1: 年龄分布（按是否患心血管疾病区分） ----
    st.subheader("年龄分布（按是否患心血管疾病区分）")
    fig_age = px.histogram(
        filtered,
        x="age_years",
        color="cardio_label",
        barmode="overlay",
        opacity=0.6,
        color_discrete_map=SERIES_COLORS,
        labels={"age_years": "年龄（岁）", "cardio_label": "心血管疾病"},
    )
    fig_age.update_layout(
        template="plotly_white",
        font=PLOTLY_FONT,
        xaxis_title="年龄（岁）",
        yaxis_title="人数",
        legend_title="心血管疾病",
        bargap=0.02,
    )
    st.plotly_chart(fig_age, width="stretch")

    # ---- 图表2: BMI 类别对心血管疾病的影响（堆叠柱状图） ----
    st.subheader("BMI 类别对心血管疾病的影响")
    fig_bmi = px.histogram(
        filtered,
        x="bmi_category",
        color="cardio_label",
        barmode="stack",
        color_discrete_map=SERIES_COLORS,
        category_orders={"bmi_category": BMI_CATEGORY_ORDER},
        labels={"bmi_category": "BMI 类别", "cardio_label": "心血管疾病"},
    )
    fig_bmi.update_layout(
        template="plotly_white",
        font=PLOTLY_FONT,
        xaxis_title="BMI 类别",
        yaxis_title="人数",
        legend_title="心血管疾病",
        bargap=0.1,
    )
    st.plotly_chart(fig_bmi, width="stretch")


if __name__ == "__main__":
    # set_page_config 必须是脚本中第一个执行的 Streamlit 命令
    st.set_page_config(
        page_title="CardioAI · 心血管疾病数据看板",
        page_icon="❤️",
        layout="wide",
    )
    df_raw = load_data(DATA_PATH)
    build_ui(df_raw)
