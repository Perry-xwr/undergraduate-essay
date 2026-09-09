import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. 设置页面基础信息 ---
st.set_page_config(page_title="工业参数优化系统", layout="wide")
st.title("🏭 基于深度学习的多变量关联预测与优化系统")
st.markdown("### Natural Language Driven Interpretable Optimization System")

# --- 2. 加载模型和工具 ---
# 定义模型结构 (必须与训练时一致)
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.model(x)

@st.cache_resource # 缓存模型，避免每次刷新都重读
def load_resources():
    device = torch.device("cpu")
    model = EnergyPredictor().to(device)
    
    # 加载模型参数
    model_path = '../saved_models/dl_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        st.error("错误：找不到模型文件 dl_model.pth")
        return None, None, None

    # 为了归一化，我们需要重新拟合一下Scaler (简单起见，直接用训练数据拟合)
    # 在实际生产中，应该保存 scaler.pkl 直接加载，这里为了演示方便重新拟合
    data_path = '../data/ENB2012_data.xlsx'
    df = pd.read_excel(data_path)
    X = df.iloc[:, :8].values
    Y = df.iloc[:, 8:].values
    
    scaler_x = MinMaxScaler().fit(X)
    scaler_y = MinMaxScaler().fit(Y)
    
    return model, scaler_x, scaler_y

model, scaler_x, scaler_y = load_resources()

# --- 3. 侧边栏：参数输入区 (模拟自然语言解析后的规格) ---
st.sidebar.header("🛠 参数配置 (Optimization Spec)")
st.sidebar.info("模拟：LLM 解析用户需求后生成的参数范围")

def user_input_features():
    # 根据数据集的范围设置滑块
    X1 = st.sidebar.slider("相对紧凑度 (Relative Compactness)", 0.6, 1.0, 0.75)
    X2 = st.sidebar.slider("表面积 (Surface Area)", 500.0, 850.0, 600.0)
    X3 = st.sidebar.slider("墙面积 (Wall Area)", 200.0, 450.0, 300.0)
    X4 = st.sidebar.slider("屋顶面积 (Roof Area)", 100.0, 250.0, 150.0) # 这是最重要的特征！
    X5 = st.sidebar.slider("总高度 (Overall Height)", 3.0, 8.0, 7.0)
    X6 = st.sidebar.selectbox("朝向 (Orientation)", [2, 3, 4, 5], index=0)
    X7 = st.sidebar.slider("玻璃面积 (Glazing Area)", 0.0, 0.4, 0.1)
    X8 = st.sidebar.slider("玻璃分布 (Glazing Area Dist)", 0, 5, 2)
    
    data = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 
            'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. 主界面：显示预测结果 ---
st.subheader("1. 实时预测 (Real-time Prediction)")

if st.button('🚀 执行预测'):
    # 预处理输入
    input_scaled = scaler_x.transform(input_df.values)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 模型推理
    with torch.no_grad():
        pred_scaled = model(input_tensor)
        pred = scaler_y.inverse_transform(pred_scaled.numpy())
    
    y1_pred = pred[0][0]
    y2_pred = pred[0][1]
    
    # 显示结果卡片
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="预测供暖负荷 (Heating Load)", value=f"{y1_pred:.2f} kWh")
    with col2:
        st.metric(label="预测制冷负荷 (Cooling Load)", value=f"{y2_pred:.2f} kWh")

# --- 5. 显示可解释性分析 ---
st.subheader("2. 证据链解释 (SHAP Evidence Chain)")
st.write("基于全量数据的特征重要性分析（已生成）：")

col3, col4 = st.columns(2)
with col3:
    st.image('../images/shap_summary_Y1.png', caption='供暖负荷的影响因子 (Top1: 屋顶面积?)', use_column_width=True)
with col4:
    st.image('../images/shap_summary_Y2.png', caption='制冷负荷的影响因子', use_column_width=True)

st.success("系统加载完毕！您现在是一个拥有 'AI 预测' + '可视化解释' 能力的工程师了。")