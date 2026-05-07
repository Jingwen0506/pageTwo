import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. 必须保留的自定义类定义 (防止加载报错) ---
class FrequencyTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.top_categories_ = {}
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# --- 2. 页面设置 ---
st.set_page_config(page_title="Bladder Cancer Recurrence Risk Predictor", layout="wide")

# --- 3. 加载模型 ---
@st.cache_resource
def load_model():
    return joblib.load("bladder_model.joblib")

model = load_model()

# --- 4. 侧边栏：输入指标 ---
st.sidebar.header("📋 Patient Clinical Indicators")

def get_input():
    st.sidebar.caption("*Note: All fields are required for accurate prediction.*")
    
    # 1. 连续数值 (Number Input)
    rbc = st.sidebar.number_input(
        "Urinary RBC Count (RBC#)", 
        min_value=0.0, 
        max_value=100.0, 
        value=5.2, 
        step=0.1,
        help="Unit: ×10⁶/L. Normal reference range is approximately 0-8."
    )

    # 2. 严格的下拉选择框 (去掉所有不详/空白选项)
    # 恶性值映射
    malig_options = {
        "High-grade": 0, 
        "Low-grade": 1, 
        "Other (Benign, PUNLMP, Atypical, Dysplasia)": 2
    }
    malig_choice = st.sidebar.selectbox("Malignancy Grade", options=list(malig_options.keys()))
    malignancy = malig_options[malig_choice]

    # 浸润值映射
    infil_options = {"Non-invasive": 0, "Invasive": 1}
    infil_choice = st.sidebar.selectbox("Infiltration Status", options=list(infil_options.keys()))
    infiltration = infil_options[infil_choice]

    # 形状值映射
    shape_options = {"Papillary": 0, "Non-papillary (Cauliflower, Nodular, etc.)": 1}
    shape_choice = st.sidebar.selectbox("Tumor Shape", options=list(shape_options.keys()))
    shape = shape_options[shape_choice]

    # 肾盂积水值映射
    hydro_options = {"Absent": 0, "Present (Unilateral/Bilateral)": 1}
    hydro_choice = st.sidebar.selectbox("Hydronephrosis Status", options=list(hydro_options.keys()))
    hydronephrosis = hydro_options[hydro_choice]

    # 坏死值映射
    necro_options = {"Absent": 0, "Present": 1}
    necro_choice = st.sidebar.selectbox("Cystic Necrosis", options=list(necro_options.keys()))
    necrosis = necro_options[necro_choice]

    # 3. 构造数据对象
    data = {
        '红细胞计数(RBC#)-尿液': rbc,
        '恶性值': malignancy,
        '浸润值': infiltration,
        '形状值': shape,
        '肾盂积水值': hydronephrosis,
        '坏死值': necrosis
    }
    return pd.DataFrame(data, index=[0])

input_df = get_input()

# --- 5. 主界面：展示结果 ---
st.title("🔬 Predictive Model for 2-Year Recurrence Risk Following Bladder Cancer Surgery")
st.write("Enter clinical characteristics in the sidebar to estimate the individualized 2-year recurrence risk.")

if st.button("Generate Prediction"):
    # 执行预测获取概率
    prob = model.predict_proba(input_df)[0, 1]
    
    st.divider()
    st.subheader(f"Predicted Recurrence Probability: {prob:.2%}")
    
    # 简单的风险分层
    if prob < 0.2:
        level, advice = "Low Risk", "Routine follow-up is recommended."
        st.success(f"Risk Level: {level}")
    elif prob < 0.4:
        level, advice = "Low-Medium Risk", "Close follow-up is recommended."
        st.info(f"Risk Level: {level}")
    elif prob < 0.6:
        level, advice = "Medium Risk", "Enhanced monitoring is recommended."
        st.warning(f"Risk Level: {level}")
    elif prob < 0.8:
        level, advice = "Medium-High Risk", "Active intervention is suggested."
        st.error(f"Risk Level: {level}")
    else:
        level, advice = "High Risk", "Immediate intervention is highly recommended."
        st.error(f"Risk Level: {level}")

st.write("")
st.info("⚠️ Disclaimer: This tool is intended for research and educational purposes only. It does not constitute medical advice or a formal clinical diagnosis.")