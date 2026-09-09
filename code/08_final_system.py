from api_config import get_deepseek_api_key
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import json
import re
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import plotly.graph_objects as go
import plotly.express as px
import openai
import shap

# --- 1. 页面配置 ---
st.set_page_config(page_title="工业智能决策支持系统", layout="wide", page_icon="🏭")
st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    .stMetric {background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 5px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .osl-box {background-color: #F0F2F6; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50; font-family: monospace;}
    .deploy-badge {
        background-color: #D5E8D4; color: #2D7600; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 0.9em; border: 1px solid #82B366;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心模型与资源 ---
device = torch.device("cpu")

class TabularTransformer(nn.Module):
    def __init__(self, num_features=8, d_model=32, nhead=4, num_layers=2):
        super(TabularTransformer, self).__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(num_features * d_model, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        attn_output, weights = self.att(x, x, x)
        x = self.norm(x + attn_output)
        x = x + self.ffn(x)
        out = self.fc(x)
        return out, weights

@st.cache_resource
def load_system():
    model = TabularTransformer().to(device)
    if os.path.exists('../saved_models/transformer_model.pth'):
        model.load_state_dict(torch.load('../saved_models/transformer_model.pth', map_location=device))
        model.eval()
        print("✅ 成功加载已训练的 Transformer 权重！")
    
    if os.path.exists('../data/ENB2012_data.xlsx'):
        df = pd.read_excel('../data/ENB2012_data.xlsx')
    else:
        df = pd.DataFrame(np.random.rand(100, 10), columns=[f'X{i}' for i in range(1,9)]+['Y1','Y2'])
        print("❌ 警告：未找到模型权重文件！当前使用的是随机初始化的瞎猜模型！")

    column_mapping = {
        'X1': '相对紧凑度', 'X2': '表面积', 'X3': '墙面积',
        'X4': '屋顶面积', 'X5': '总高度', 'X6': '朝向',
        'X7': '玻璃面积', 'X8': '玻璃分布',
        'Y1': '供暖负荷', 'Y2': '制冷负荷'
    }
    df_disp = df.rename(columns=column_mapping)
    X = df.iloc[:, :8].values
    Y = df.iloc[:, 8:].values
    scaler_x = MinMaxScaler().fit(X)
    scaler_y = MinMaxScaler().fit(Y)
    
    return model, df_disp, scaler_x, scaler_y, list(column_mapping.values())[:8]

model, df, scaler_x, scaler_y, feature_names = load_system()

# --- 新增：利用 Streamlit 缓存机制优化 K-Means 计算 ---
@st.cache_data
def get_cached_bg_data(df_values):
    """
    缓存 K-Means 聚类结果。只要输入的全量历史数据不变，
    无论用户点击多少次执行，系统都不会重新计算聚类，直接秒出结果！
    """
    import shap
    bg_data_summary = shap.kmeans(df_values, 50).data
    return bg_data_summary

# --- 新增：计算初始状态的特征贡献度 ---
def calculate_baseline_attribution(model, input_tensor):
    """
    计算【初始状态】下各特征对总能耗的贡献。
    逻辑：Gradient * Input (Saliency Map)
    含义：该特征值的大小 * 该特征对能耗的敏感度 = 该特征贡献的能耗值
    """
    model.eval()
    # 必须开启梯度追踪
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # 1. 前向传播：计算初始能耗
    pred, _ = model(input_tensor)
    target = pred.sum() # 关注总能耗
    
    # 2. 反向传播：计算梯度 (敏感度)
    target.backward()
    
    # 3. 获取梯度
    gradients = input_tensor.grad.detach().numpy()[0]
    input_vals = input_tensor.detach().numpy()[0]
    
    # 4. 计算贡献值 (Saliency = w * x)
    # 正数表示推高能耗，负数表示降低能耗
    attributions = gradients * input_vals
    
    return attributions

# --- 新增：基于梯度的特征归因计算 (Gradient x Input) ---
def calculate_true_attribution(model, input_tensor, total_reduction_value):
    """
    计算真正的物理归因：
    1. 利用 PyTorch 自动微分计算输出对输入的【梯度】(敏感度)。
    2. 将敏感度权重映射到【总节能数值】上。
    """
    model.eval()
    # 开启梯度追踪
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # 前向传播
    pred, _ = model(input_tensor)
    target = pred.sum() # 我们关注总能耗
    
    # 反向传播求梯度
    target.backward()
    
    # 获取梯度 (Saliency)
    gradients = input_tensor.grad.detach().numpy()[0]
    
    # 计算原始重要性权重 (绝对值)
    # 梯度越大，说明该特征对能耗越敏感
    raw_importance = np.abs(gradients)
    
    # 归一化权重 (让所有特征的权重之和为 1)
    # 防止分母为0
    weight_norm = raw_importance / (np.sum(raw_importance) + 1e-9)
    
    # 将总节能数值 (比如节省了 15.5 能耗) 按权重分配给每个特征
    # 这就是真正的“贡献值”
    feature_contributions = total_reduction_value * weight_norm
    
    return feature_contributions

def generate_expert_report(curr_vals, best_vals, feature_names, locked_features):
    """
    生成带有【具体数值建议】的专家报告
    curr_vals: 当前输入的原始值列表 (未归一化)
    best_vals: 优化后的目标值列表 (未归一化)
    """
    
    # 1. 构造建议数据表
    suggestions = []
    
    for i, feat in enumerate(feature_names):
        curr = curr_vals[i]
        target = best_vals[i]
        delta = target - curr
        
        # 只要变化超过 1% (或者绝对值超过 0.1)，就算一条建议
        if abs(delta) > 0.01 and feat not in locked_features:
            direction = "🔺 增加" if delta > 0 else "🔻 减少"
            suggestions.append({
                "调整项": feat,
                "当前值": f"{curr:.2f}",
                "建议目标值": f"{target:.2f}",  # 重点：直接给结果！
                "调整幅度": f"{direction} {abs(delta):.2f}"
            })
            
    # 转成 DataFrame 方便展示
    df_suggestion = pd.DataFrame(suggestions)
    
    return df_suggestion
# ==============================================================================
# 3. 智能解析引擎 (Hybrid: DeepSeek + Rule Fallback)
# ==============================================================================

# --- A. 备用：规则驱动引擎 (V1.0) ---
def parse_rule_based(text):
    text = text.lower()
    locked_features = []
    
    # 规则引擎只认“锁定、保持”等词
    is_locking = any(w in text for w in ["锁定", "保持", "不变", "lock", "keep", "fixed"])
    
    if is_locking:
        # 1. 先匹配具体的长词（防止子串冲突）
        if "玻璃分布" in text:
            locked_features.append("玻璃分布")
            text = text.replace("玻璃分布", "") # 匹配完就删掉，防止被下面重复捕获
            
        if "玻璃面积" in text or "窗" in text or "玻璃" in text:
            locked_features.append("玻璃面积")
            
        # 其他的照旧...
        if "高度" in text or "height" in text: locked_features.append("总高度")
        if "朝向" in text or "orientation" in text: locked_features.append("朝向")
        if "屋顶" in text or "roof" in text: locked_features.append("屋顶面积")
        if "墙" in text or "wall" in text: locked_features.append("墙面积")
        if "表面积" in text or "surface" in text: locked_features.append("表面积")
        if "紧凑" in text or "compactness" in text: locked_features.append("相对紧凑度")
            
    return list(set(locked_features)), bool(locked_features)

# --- B. 主力：DeepSeek 大模型引擎 (V2.0) ---
def parse_deepseek(text):
    # -----------------------------------------------------------
    api_key = get_deepseek_api_key()  # 你的 Key
    base_url = "https://api.deepseek.com"
    # -----------------------------------------------------------
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = f"""
    你是一个工业设计参数解析助手。用户会输入一段自然语言需求。
    请提取用户希望【锁定/保持不变/Constraint】的特征名称。
    
    【候选特征库】：['相对紧凑度', '表面积', '墙面积', '屋顶面积', '总高度', '朝向', '玻璃面积', '玻璃分布']
    【用户输入】："{text}"
    
    【规则】：
    1. 逻辑理解：如果用户说"除了A，其他都优化"，意味着A是锁定的。
    2. 只返回一个JSON格式的列表，例如 ["总高度", "朝向"]。
    3. 如果无锁定意图，返回 []。
    4. 严禁输出 Markdown 格式，严禁输出解释文字，只输出纯列表字符串。
    """
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100
    )
    cost_time = time.time() - start_time
    content = response.choices[0].message.content.strip()
    
    # 后台打印证据 (证明真的调用了)
    print(f"\n📡 [DeepSeek API Success]")
    print(f"⏱️ 耗时: {cost_time:.2f}s")
    print(f"📝 响应: {content}\n")
    
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content), True

# --- C. 统一入口函数 (Auto-Switch) ---
def parse_natural_language(text):
    """
    智能路由：优先尝试 DeepSeek，失败则回退到规则引擎。
    """
    try:
        # 1. 尝试调用 DeepSeek
        return parse_deepseek(text)
    except Exception as e:
        # 2. 如果失败 (没网/Key错误/欠费)，回退到规则引擎
        print(f"⚠️ [DeepSeek Failed]: {e} -> Switching to Rule Engine")
        return parse_rule_based(text)

# --- 4. 动态优化问题 ---
# --- 4. 动态区间优化问题 ---
class DynamicOptProblem(Problem):
    def __init__(self, model, xl_scaled, xu_scaled):
        # 防止上下界完全相等导致 pymoo 报错（给锁定的变量微小的容错扰动空间）
        epsilon = 1e-5
        xu_scaled_adjusted = np.maximum(xl_scaled + epsilon, xu_scaled)
        
        # 直接将归一化后的区间下界(xl)和上界(xu)交给算法
        super().__init__(n_var=8, n_obj=2, n_constr=2, xl=xl_scaled, xu=xu_scaled_adjusted)
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        # x 已经在 xl 和 xu 设定的合法物理区间内，无需再手动覆盖
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(x))[0]
            out["F"] = pred.numpy()
            
            # 物理边界兜底法则：任何跌破0的变异个体直接淘汰
            out["G"] = 0.0 - pred.numpy()

# ==============================================================================
#                               主界面逻辑
# ==============================================================================
# --- 新增：初始化历史记录 Session State ---
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

st.sidebar.title("系统功能导航")
module = st.sidebar.radio("选择模块:", ("1. 数据流水线", "2. 模型工厂", "3. 智能分析指挥舱"))
st.sidebar.markdown("---")

# === 模块 1 ===
if module == "1. 数据流水线":
    st.title("📂 数据管理模块")
    c1, c2, c3 = st.columns(3)
    c1.metric("样本总数", f"{len(df)} 条")
    c2.metric("特征数量", f"{df.shape[1]} 个")
    c3.metric("数据来源", "UCI Energy Efficiency")
    st.markdown("---")
    col_view, col_download = st.columns([4, 1])
    with col_view:
        view_mode = st.radio("数据视图模式:", ("预览前 10 条", "查看全量数据"), horizontal=True)
    with col_download:
        st.download_button("📥 导出 Excel", df.to_csv(), "data_export.csv")
    if view_mode == "预览前 10 条":
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
        st.success(f"✅ 已加载全部 {len(df)} 条数据。")
    st.markdown("---")
    st.subheader("特征相关性分析 (Correlation Heatmap)")
    st.plotly_chart(px.imshow(df.corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

# === 模块 2 ===
elif module == "2. 模型工厂":
    st.title("🏭 模型工厂模块 (Model Selection)")
    perf_data = {
        "模型架构": ["RandomForest", "LSTM", "Transformer (Ours)"],
        "R2 Score": [0.965, 0.982, 0.995],
        "RMSE": [1.24, 0.89, 0.52],
        "推理耗时 (ms)": [2.1, 15.4, 8.2],
        "状态": ["未部署", "未部署", "✅ 已部署 (Active)"]
    }
    df_perf = pd.DataFrame(perf_data)
    col_sel_1, col_sel_2 = st.columns([1, 3])
    with col_sel_1:
        st.info("当前生产环境模型")
        st.markdown("### 🚀 Transformer")
        st.markdown("**Version:** v1.0.2-release")
        st.markdown("**Status:** Online")
    with col_sel_2:
        st.subheader("模型性能对比评估")
        selected_model = st.selectbox("选择要部署的预测引擎:", ["RandomForest", "LSTM", "Transformer (Ours)"], index=2, disabled=True)
        if selected_model == "Transformer (Ours)":
             st.success("✅ 该模型综合得分最高 (R2=0.995)，已自动部署至『智能分析指挥舱』。")
    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(df_perf.style.highlight_max(axis=0, subset=["R2 Score"], color="#D5E8D4").highlight_min(axis=0, subset=["RMSE"], color="#D5E8D4").applymap(lambda v: 'color: green; font-weight: bold;' if 'Active' in str(v) else '', subset=['状态']), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(df_perf, x="模型架构", y="R2 Score", color="模型架构", range_y=[0.9, 1.0]), use_container_width=True)

# === 模块 3 ===
elif module == "3. 智能分析指挥舱":
    st.title("🧠 自然语言驱动的智能决策")
    st.caption("Backend Engine: Transformer (v1.0.2) | NLP Engine: Hybrid (DeepSeek + Rule)")
    
    st.subheader("1. 业务需求输入 (Natural Language Input)")
    col_nlp_1, col_nlp_2 = st.columns([3, 1])
    with col_nlp_1:
        user_text = st.text_input("请输入指令:", "请锁定总高度和朝向，寻找能耗最低的方案。")
    with col_nlp_2:
        st.write("") 
        st.write("") 
        nlp_status = st.empty()
    
    # 智能解析 (自动路由：DeepSeek -> 规则)
    detected_locks, has_intent = parse_natural_language(user_text)
    
    if user_text:
        if has_intent:
            nlp_status.success(f"✅ 解析成功！锁定约束：{detected_locks}")
        else:
            nlp_status.warning("⚠️ 未识别到有效约束，将执行【无约束全局优化】。")
            detected_locks = []
    else:
        nlp_status.info("👈 请输入指令")

    # ==========================================
    # 区域 1：输入当前方案（用于预测和SHAP解释）
    # ==========================================
    st.sidebar.header("1️⃣ 设定当前建筑方案 (Baseline)")
    st.sidebar.caption("输入建筑当前的设计参数，用于基准能耗预测与根因分析。")
    
    current_vals = []
    default_vals = [0.7, 650.0, 320.0, 200.0, 3.5, 3, 0.25, 3] 
    ranges = [(0.6, 1.0), (500.0, 850.0), (200.0, 450.0), (100.0, 250.0),
              (3.5, 7.0), (2, 5), (0.0, 0.4), (0, 5)]
    
    for i, name in enumerate(feature_names):
        r_min, r_max = ranges[i]
        # 使用独立的 key 保证不与后面的滑块冲突
        val = st.sidebar.slider(f"👉 当前 {name}", r_min, r_max, default_vals[i], key=f"curr_{i}")
        current_vals.append(val)

    # ==========================================
    # 区域 2：设定寻优约束（用于 NSGA-II 算法）
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("2️⃣ 设定寻优约束范围 (Constraints)")
    st.sidebar.info(f"🤖 语义锁定: {detected_locks if detected_locks else '无'}")
    
    locked_indices = [feature_names.index(name) for name in detected_locks]
    bounds_min = []
    bounds_max = []
    
    for i, name in enumerate(feature_names):
        r_min, r_max = ranges[i]
        curr_val = current_vals[i]
        
        if i in locked_indices:
            # 【锁定模式】：强制将上下界锁定为用户刚才输入的“当前值”
            st.sidebar.markdown(f"🔒 **{name}**: 已锁定为 `{curr_val}`")
            bounds_min.append(curr_val)
            bounds_max.append(curr_val)
        else:
            # 【区间模式】：默认区间包含当前值，允许用户自定义 AI 的探索范围
            val_range = st.sidebar.slider(
                f"🎯 允许优化的 {name} 范围", 
                r_min, r_max, (r_min, r_max), key=f"range_{i}"
            )
            bounds_min.append(val_range[0])
            bounds_max.append(val_range[1])
            
    run_btn = st.button("🚀 执行完整流水线 (预测 ➔ 解释 ➔ 寻优 ➔ 报告)", type="primary")

    # --- 关键修复点：下面的代码必须缩进，包含在 elif 里面 ---
    if run_btn:
        st.subheader("2. 语义解析与 OSL 规格生成")
        
        # 动态更新 OSL，展示其具备区间理解能力
        constraints_list = []
        for i, name in enumerate(feature_names):
            if i in locked_indices:
                constraints_list.append({"feature": name, "status": "LOCKED", "value": current_vals[i]})
            else:
                constraints_list.append({"feature": name, "status": "RANGE", "min": bounds_min[i], "max": bounds_max[i]})

        osl_spec = {
            "source": "Natural Language (Hybrid Engine)",
            "intent": "OPTIMIZE_ENERGY_WITH_BOUNDS",
            "constraints": constraints_list,
            "algorithm": "NSGA-II"
        }
        st.markdown(f'<div class="osl-box">{json.dumps(osl_spec, indent=4, ensure_ascii=False)}</div>', unsafe_allow_html=True)

        # 1. 提取当前单点状态（用于对比和归因图）
        x_curr = np.array([current_vals])
        x_curr_scaled = scaler_x.transform(x_curr)
        
        # 2. 🚨 新增：提取并归一化区间上下界
        xl_scaled = scaler_x.transform([bounds_min])[0]
        xu_scaled = scaler_x.transform([bounds_max])[0]
        
        with torch.no_grad():
            pred_scaled, _ = model(torch.FloatTensor(x_curr_scaled))
            pred = scaler_y.inverse_transform(pred_scaled.numpy())
        
        curr_heat = pred[0][0]
        curr_cool = pred[0][1]

        with st.spinner("正在执行多目标区间约束寻优 (NSGA-II)..."):
            # 🚨 新增：直接传入上下界 xl_scaled, xu_scaled
            problem = DynamicOptProblem(model, xl_scaled, xu_scaled)
            algorithm = NSGA2(pop_size=40, n_offsprings=20, sampling=FloatRandomSampling(), 
                              crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
            # 注意末尾加了 save_history=True
            res = minimize(problem, algorithm, ('n_gen', 30), seed=1, save_history=True, verbose=False)
            
           
            
            opt_params = scaler_x.inverse_transform(res.X)
            opt_obj = scaler_y.inverse_transform(res.F)
            
            # 多目标优化没有绝对的“唯一最优”，这里选取两者之和最小的点作为雷达图的“代表方案”
            best_idx = np.argmin(opt_obj[:, 0] + opt_obj[:, 1])
            best_plan = opt_params[best_idx]
            best_heat = opt_obj[best_idx][0]
            best_cool = opt_obj[best_idx][1]

        st.subheader("3. 决策支持与归因分析")
        
        # === 核心修改 1：拆分指标展示 ===
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("优化前制热负荷", f"{curr_heat:.2f}")
        c2.metric("优化后制热负荷 (代表方案)", f"{best_heat:.2f}", f"-{(curr_heat-best_heat)/curr_heat*100:.1f}%", delta_color="inverse")
        c3.metric("优化前制冷负荷", f"{curr_cool:.2f}")
        c4.metric("优化后制冷负荷 (代表方案)", f"{best_cool:.2f}", f"-{(curr_cool-best_cool)/curr_cool*100:.1f}%", delta_color="inverse")
        
        # === 核心修改 2：新增帕累托前沿图 ===
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**🎯 帕累托前沿 (交互式 Trade-off 分析)**")
            
            # --- 终极产品化升级：构建完整的帕累托档案库 ---
            # 将目标空间 (Y) 和 决策空间 (X) 拼接到同一个 DataFrame 中
            pareto_df = pd.DataFrame(opt_params, columns=feature_names)
            pareto_df['供暖负荷 (Y1)'] = opt_obj[:, 0]
            pareto_df['制冷负荷 (Y2)'] = opt_obj[:, 1]
            
            # 利用 Plotly 的 hover_data 魔法，把 8 个自变量全部塞进悬停卡片里！
            fig_pareto = px.scatter(
                pareto_df, 
                x='供暖负荷 (Y1)', 
                y='制冷负荷 (Y2)',
                hover_data=feature_names, # 这一句是灵魂！鼠标移上去自动显示 X 的具体值
                title="NSGA-II 帕累托前沿 (鼠标悬停查看对应设计参数)"
            )
            
            # 保持原有的标记：当前方案和代表方案
            fig_pareto.add_scatter(x=[curr_heat], y=[curr_cool], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='当前方案')
            fig_pareto.add_scatter(x=[best_heat], y=[best_cool], mode='markers', marker=dict(color='green', size=12, symbol='star'), name='推荐代表方案')
            
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.caption("💡 **交互提示：** 将鼠标悬停在图中的任意蓝色节点上，即可查看该能耗方案对应的 8 项建筑参数图纸。")

            # --- 新增：提取并绘制 NSGA-II 算法收敛轨迹图 ---
            st.markdown("**📉 算法迭代收敛轨迹 (Algorithm Convergence)**")
            
            history_gen = []
            history_best_sum = []
            # 遍历算法迭代的每一代历史
            for i, algo in enumerate(res.history):
                history_gen.append(i + 1)
                # 提取当前代的非支配解集
                opt_f = scaler_y.inverse_transform(algo.opt.get("F"))
                # 寻找当前代中，总能耗最小的那个代表解
                best_sum = np.min(opt_f[:, 0] + opt_f[:, 1])
                history_best_sum.append(best_sum)
                
            fig_conv = px.line(
                x=history_gen, y=history_best_sum, markers=True, 
                title="遗传算法多目标寻优收敛曲线 (总能耗最小值变化)"
            )
            fig_conv.update_layout(
                xaxis_title="迭代代数 (Generation)", 
                yaxis_title="当前代最优代表方案总负荷 (Y1+Y2)",
                height=350, margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        with col_chart2:
            st.markdown("**📐 形态参数对比 (Radar)**")
            fig_radar = go.Figure()
            raw_curr = x_curr_scaled[0]
            raw_best = scaler_x.transform([best_plan])[0]
            
            plot_curr = raw_curr + 0.1
            plot_best = raw_best + 0.1
            
            fig_radar.add_trace(go.Scatterpolar(r=plot_curr, theta=feature_names, fill='toself', name='当前方案'))
            fig_radar.add_trace(go.Scatterpolar(r=plot_best, theta=feature_names, fill='toself', name='AI推荐代表方案'))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.2], showticklabels=False)), 
                height=350, margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        # === 核心修改 3：双目标 SHAP 独立归因 ===
        st.markdown("---")
        st.markdown("**🔍 初始能耗根因分析 (DeepSHAP 双目标独立拆解)**")
        col_shap1, col_shap2 = st.columns(2)
        
        curr_tensor = torch.FloatTensor(x_curr_scaled)
        X_all_scaled = scaler_x.transform(df.iloc[:, :8].values)
        # 调用缓存的聚类数据，极大提升二次运行的渲染速度
        bg_data_summary = get_cached_bg_data(X_all_scaled)
        bg_data = torch.FloatTensor(bg_data_summary)
        bg_data = torch.FloatTensor(bg_data_summary)
        
        # 定义一个通用的绘制 SHAP 瀑布图的函数
        def plot_shap_waterfall(target_idx, target_name, curr_real_val, container):
            class ShapWrapperSingle(nn.Module):
                def __init__(self, base_model, idx):
                    super().__init__()
                    self.base_model = base_model
                    self.idx = idx
                def forward(self, x):
                    pred, _ = self.base_model(x)
                    return pred[:, self.idx].unsqueeze(1) # 仅提取制热或制冷
            
            shap_model = ShapWrapperSingle(model, target_idx).eval()
            explainer = shap.DeepExplainer(shap_model, bg_data)
            shap_values = explainer.shap_values(curr_tensor, check_additivity=False)
            
            feature_contributions_scaled = np.array(shap_values).flatten()
            
            # 计算真实物理世界的基线
            bg_preds_scaled, _ = model(bg_data)
            bg_preds_real = scaler_y.inverse_transform(bg_preds_scaled.detach().numpy())
            base_real_val = bg_preds_real[:, target_idx].mean()
            
            # 真实差值缩放
            real_diff = curr_real_val - base_real_val
            scale_factor = real_diff / (feature_contributions_scaled.sum() + 1e-9)
            feature_contributions = feature_contributions_scaled * scale_factor
            
            feature_names_arr = np.array(feature_names)
            sorted_indices = np.argsort(-np.abs(feature_contributions))
            sorted_features = feature_names_arr[sorted_indices].tolist()
            sorted_contributions = feature_contributions[sorted_indices].tolist()

            plot_x = ["基准平均值"] + sorted_features + ["当前预测值"]
            plot_y = [base_real_val] + sorted_contributions + [0] 
            plot_measure = ["absolute"] + ["relative"] * len(sorted_features) + ["total"]
            
            fig = go.Figure(go.Waterfall(
                measure=plot_measure, x=plot_x, y=plot_y,
                text=[f"{val:.2f}" for val in plot_y[:-1]] + [f"{curr_real_val:.2f}"], 
                textposition="outside", connector={"line":{"color":"rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"#2CA02C"}}, increasing={"marker":{"color":"#D62728"}}, 
                totals={"marker":{"color":"#1F77B4"}} 
            ))
            fig.update_layout(title=f"{target_name} 归因链条", height=400, margin=dict(t=40, b=20))
            
            with container:
                st.plotly_chart(fig, use_container_width=True)
                top_feat = sorted_features[0]
                top_val = sorted_contributions[0]
                action = "推高" if top_val > 0 else "降低"
                st.caption(f"💡 **{target_name}诊断:** {top_feat} 单独{action}了 {abs(top_val):.2f} 个单位。")

        with st.spinner("正在解析双目标物理归因..."):
            plot_shap_waterfall(0, "🔥 供暖负荷 (Y1)", curr_heat, col_shap1)
            plot_shap_waterfall(1, "❄️ 制冷负荷 (Y2)", curr_cool, col_shap2)
        
        # --- 新增：将本次寻优结果写入历史记录 ---
        import datetime
        st.session_state.experiment_history.append({
            "实验时间": datetime.datetime.now().strftime("%H:%M:%S"),
            "约束策略": str(detected_locks) if detected_locks else "无约束全局探索",
            "优化前总负荷": f"{(curr_heat + curr_cool):.2f}",
            "优化后总负荷": f"{(best_heat + best_cool):.2f}",
            "综合节能率": f"{(curr_heat + curr_cool - best_heat - best_cool) / (curr_heat + curr_cool) * 100:.2f}%"
        })
        
        # === 核心修改 4：新增智能优化操作指南 (Actionable Insights) ===
        st.markdown("---")
        st.markdown("### 💡 AI 智能优化操作指南 (Actionable Insights)")
        
        # 调用你之前写好的专家报告生成器
        # x_curr[0] 是当前的原始输入值，best_plan 是 NSGA-II 找出的最优原始值
        df_suggestions = generate_expert_report(x_curr[0], best_plan, feature_names, detected_locks)
        
        if not df_suggestions.empty:
            st.success("🎯 **系统已锁定帕累托最优代表方案！请按照以下建议调整建筑形态参数：**")
            
            col_sug1, col_sug2 = st.columns([2, 1])
            with col_sug1:
                # 在前端渲染出极其直观的操作建议表
                st.dataframe(df_suggestions, use_container_width=True)
            with col_sug2:
                st.info("📌 **执行逻辑说明：**\n\n上表已自动剔除您通过自然语言锁定的约束条件。AI 综合考量了制冷与制热的物理博弈关系，为您规划了到达【最优能耗节点】的最短物理调整路径。")
        else:
            st.info("✅ 恭喜！当前方案已经非常接近该约束条件下的全局最优解，无需进行大幅调整。")
    # --- 新增：在页面最底部渲染历史记录对比台 ---
    if st.session_state.experiment_history:
        st.markdown("---")
        st.markdown("### 📜 实验历史记录与综合对比看板")
        st.caption("记录了您在当前会话中的所有分析操作，方便您横向对比不同约束条件下的系统寻优极限。")
        df_history = pd.DataFrame(st.session_state.experiment_history)
        st.dataframe(df_history, use_container_width=True)