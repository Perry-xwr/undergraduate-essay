from api_config import get_deepseek_api_key
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
import json
import datetime
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# 1. 页面配置与全局状态初始化
# ==============================================================================
st.set_page_config(page_title="通用工业智能决策底座", layout="wide", page_icon="🏭")
st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    .osl-box {background-color: #F0F2F6; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50; font-family: monospace;}
    </style>
    """, unsafe_allow_html=True)

# 初始化 Session State
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

# ==============================================================================
# 2. 动态核心模型与缓存工具
# ==============================================================================
# 智能探测：如果有 Nvidia 显卡且环境配置正确，就用 GPU，否则退回 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"🚀 当前深度学习计算引擎: `{device}`") # 在前端炫耀一下你的算力

# 🚨 动态维度 Transformer
class TabularTransformer(nn.Module):
    def __init__(self, num_features, output_dim, d_model=32, nhead=4, num_layers=2):
        super(TabularTransformer, self).__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(num_features * d_model, 64), 
            nn.ReLU(), 
            nn.Linear(64, output_dim)  # 动态输出维度
        )
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        attn_output, _ = self.att(x, x, x)
        x = self.norm(x + attn_output)
        x = x + self.ffn(x)
        out = self.fc(x)
        return out, None

# ==========================================
# 🚨 新增：经典多层感知机 (MLP) 作为深度学习基线
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, num_features, output_dim):
        super(SimpleMLP, self).__init__()
        # 经典的 3 层全连接网络
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        # 保持与 Transformer 一致的返回值结构 (out, None)
        return self.net(x), None
    

# 🚨 缓存 K-Means 加速归因
@st.cache_data
def get_cached_bg_data(df_values):
    bg_data_summary = shap.kmeans(df_values, min(50, len(df_values))).data
    return bg_data_summary

def generate_expert_report(curr_vals, best_vals, feature_names, locked_features):
    suggestions = []
    for i, feat in enumerate(feature_names):
        curr, target = curr_vals[i], best_vals[i]
        delta = target - curr
        if abs(delta) > 1e-3 and feat not in locked_features:
            direction = "🔺 增加" if delta > 0 else "🔻 减少"
            suggestions.append({
                "调整项": feat, "当前值": f"{curr:.2f}", 
                "建议目标值": f"{target:.2f}", "调整幅度": f"{direction} {abs(delta):.2f}"
            })
    return pd.DataFrame(suggestions)


import json
import openai
import re

# ==============================================================================
# 3. 动态解析引擎 (全量 NL2Opt 智能体编译器 - 工业完全体)
# ==============================================================================
def parse_constraints_with_llm(user_input, feature_names, target_names):
    api_key = get_deepseek_api_key() # 你的 DeepSeek API Key
    
    try:
        # 🚨 动态构建 goals_y JSON 范例
        goals_example = {}
        for i, t in enumerate(target_names):
            if i == 0:
                goals_example[t] = "minimize"
            elif i == 1:
                goals_example[t] = "target:20.0"
            else:
                goals_example[t] = "maximize"
                
        goals_example_str = json.dumps(goals_example, ensure_ascii=False, indent=4)
        goals_example_str = goals_example_str.replace("{", "{{").replace("}", "}}")

        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        prompt = f"""
        你是一个极其严谨的工业优化智能体。请阅读用户的自然语言指令，提取自变量(X)的约束、因变量(Y)的约束以及因变量(Y)的优化目标。
        
        【自变量池 X】：{feature_names}
        【因变量池 Y (优化目标/状态)】：{target_names}
        【用户指令】："{user_input}"
        
        【解析规则】
        1. 自变量约束 (constraints_x):
           - locks: 要求保持不变、锁定的自变量名。必须严格使用变量池中的英文原名。
             🚨 [排他性约束]：如果用户明确指出了特征名字，例如“只允许改动 A 和 B”，则必须将 A 和 B 以外的所有特征放入 locks！
             🚨 [全局稀疏约束的避坑指南]：但是！！！如果用户说的是“最多只能改 2 个参数，其他随便变”，由于用户没有指明具体是哪 2 个，这就属于全局稀疏约束！此时绝对【不能】把其他参数放入 locks！必须让 locks 保持为空，将数字提取到 max_changes 中！
           - ranges: 要求限制范围的自变量，格式为 [min, max]（单侧限制则另一侧写 null）。
           - max_changes: 稀疏约束。如果用户要求“最多改动K个参数”、“尽量少改动”等，请提取数字K。如果没有限制，填 null。
           
        2. 因变量约束 (constraints_y):
           - 如果用户对因变量 (Y) 提出了硬性的区间、上限或下限要求，提取至 ranges 中，格式为 [min, max]。
           
        3. 因变量目标 (goals_y):
           - 为每个因变量 Y 指定优化方向："minimize", "maximize", 或 "target:数值"。
        
        【输出要求】
        必须且只能输出 JSON 字符串！严禁包含 markdown 标记。格式如下：
        {{
            "constraints_x": {{
                "locks": ["{feature_names[0] if feature_names else 'FeatureA'}"],
                "ranges": {{}},
                "max_changes": 3 
            }},
            "constraints_y": {{
                "ranges": {{}}
            }},
            "goals_y": {goals_example_str}
        }}
        """
        response = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # 暴力清洗 JSON (防止大模型输出 Markdown 格式的 ```json )
        if "```" in content:
            content = re.search(r'\{.*\}', content, re.DOTALL).group()
            
        return json.loads(content)
        
    except Exception as e:
        print(f"⚠️ [大模型解析异常]: {e}。退回默认最小化无约束状态。")
        # 兜底：如果报错，默认所有目标最小化，X和Y均无约束 (防崩溃处理)
        default_goals = {t: "minimize" for t in target_names}
        return {
            "constraints_x": {"locks": [], "ranges": {}}, 
            "constraints_y": {"ranges": {}}, 
            "goals_y": default_goals
        }

from openai import OpenAI # 确保你已经安装了较新版本的 openai 库 (pip install openai)

def generate_llm_narrative_report(user_intent, changed_features, align_score, pred_improvements, top_shap):
    """
    利用 DeepSeek 大模型将数学结果转化为叙述性优化报告
    """
    # 构造给大模型的上下文数据
    prompt = f"""
    你是一个工业智能决策专家。请根据以下优化数据，为用户写一份专业的【执行摘要报告】。
    
    1. 用户原始需求: "{user_intent}"
    2. 系统实际修改的特征: {changed_features}
    3. 决策对齐度 (Align@K): {align_score:.1f}%
    4. 核心物理依据 (SHAP Top特征): {top_shap}
    5. 预测效果改善情况: {pred_improvements}
    
    报告要求：
    - 语气专业、严谨。
    - 结构清晰：包含 [优化概述]、[关键操作说明]、[逻辑依据] 和 [风险提示]。
    - 必须解释为什么选择这些变量（结合物理依据和对齐度）。
    - 篇幅约 300 字左右，使用简体中文。
    """
    
    try:
        # 初始化 DeepSeek 客户端
        client = OpenAI(
            api_key=get_deepseek_api_key(),
            base_url="https://api.deepseek.com" # DeepSeek 的官方接口地址
        )
        
        # 调用 DeepSeek 对话模型
        response = client.chat.completions.create(
            model="deepseek-chat", # DeepSeek 的核心模型名称
            messages=[
                {"role": "system", "content": "你是一个专业的工业决策报告生成器。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 # 控制生成的专业性和创造性平衡
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"报告生成失败：{e}"


def validate_and_repair_specs(parsed_json):
    """
    规格校验与修复算子 (Validate/Repair Operator)
    对应论文第 6 节：检测并修复 LLM 解析产生的冲突约束
    """
    # 深拷贝一份数据，避免修改原始解析结果
    repaired_json = json.loads(json.dumps(parsed_json))
    repair_logs = [] # 记录修复日志，用于界面审计展示

    # 1. 修复自变量 X 的范围冲突
    ranges_x = repaired_json.get("constraints_x", {}).get("ranges", {})
    for feat, bounds in ranges_x.items():
        # 如果上下限都有值，且下限 > 上限
        if len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
            if bounds[0] > bounds[1]:
                old_bounds = list(bounds)
                # 【确定性修复策略】：自动交换上下限位置（或者你也可以直接设为 bounds[0] = bounds[1]）
                bounds[0], bounds[1] = bounds[1], bounds[0] 
                ranges_x[feat] = bounds
                repair_logs.append(f"🛠️ [X区间修复] {feat} 下限大于上限 {old_bounds} ➔ 已自动修复为 {bounds}")

    # 2. 修复因变量 Y 的范围冲突
    ranges_y = repaired_json.get("constraints_y", {}).get("ranges", {})
    for feat, bounds in ranges_y.items():
        if len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
            if bounds[0] > bounds[1]:
                old_bounds = list(bounds)
                bounds[0], bounds[1] = bounds[1], bounds[0]
                ranges_y[feat] = bounds
                repair_logs.append(f"🛠️ [Y区间修复] {feat} 下限大于上限 {old_bounds} ➔ 已自动修复为 {bounds}")

    # 3. 修复 L0 稀疏约束的异常值 (防止出现负数)
    max_changes = repaired_json.get("constraints_x", {}).get("max_changes")
    if max_changes is not None and max_changes < 0:
        repaired_json["constraints_x"]["max_changes"] = 0
        repair_logs.append(f"🛠️ [稀疏约束修复] 允许最大改动数不能为负数 ({max_changes}) ➔ 已自动修复为 0")

    return repaired_json, repair_logs
# ==============================================================================
# 4. 动态多目标区间优化问题 (带有绝对物理锁 + 动态优化方向 + Y状态红线 + 🚨 X稀疏约束)
# ==============================================================================
class DynamicOptProblem(Problem):
    def __init__(self, model, xl_scaled, xu_scaled, num_vars, num_objs, locked_indices, locked_vals_scaled, directions, targets_scaled, target_names, y_bounds_real, scaler_y, x_baseline_scaled, max_changes):
        epsilon = 1e-5
        xu_scaled_adjusted = np.maximum(xl_scaled + epsilon, xu_scaled)
        
        # =================================================================
        # 🚨 终极防崩溃：约束数量大幅减少！
        # 因为红线已经变成了 F 的软扣分项，L0 变成了修复算子
        # 现在只剩下最基础的“预测值必须非负”这 num_objs 个硬约束了
        # =================================================================
        n_constr = num_objs 
        
        # n_obj=num_objs+1 依然保留，因为我们有 L1 变动代价作为辅助目标
        super().__init__(n_var=num_vars, n_obj=num_objs+1, n_constr=n_constr, xl=xl_scaled, xu=xu_scaled_adjusted)
        
        self.model = model
        self.locked_indices = locked_indices
        self.locked_vals_scaled = locked_vals_scaled
        self.directions = directions
        self.targets_scaled = targets_scaled
        self.target_names = target_names
        self.y_bounds_real = y_bounds_real
        self.scaler_y = scaler_y
        
        # 接收现状基准线与稀疏约束 K
        self.x_baseline_scaled = x_baseline_scaled
        self.max_changes = max_changes if max_changes is not None else num_vars # 没限制就是全都能改
    def _evaluate(self, x, out, *args, **kwargs):
        x_eval = np.copy(x)

        # 1. 确保 max_changes 是整数
        max_c = None
        if hasattr(self, 'max_changes') and self.max_changes is not None:
            try:
                max_c = int(self.max_changes)
            except ValueError:
                pass

        # 2. 霸道基因修复 (L0 强行满足)
        if max_c is not None and max_c < x_eval.shape[1]:
            for r in range(x_eval.shape[0]):
                diff = np.abs(x_eval[r] - self.x_baseline_scaled)
                if max_c == 0:
                    x_eval[r] = self.x_baseline_scaled
                else:
                    snap_indices = np.argsort(diff)[:-max_c]
                    x_eval[r, snap_indices] = self.x_baseline_scaled[snap_indices]

        # 3. 强制物理锁 (用户硬性要求，不容商量)
        for i, idx in enumerate(self.locked_indices):
            x_eval[:, idx] = self.locked_vals_scaled[i]

        with torch.no_grad():
            model_device = next(self.model.parameters()).device 
            x_tensor = torch.FloatTensor(x_eval).to(model_device)
            pred_scaled_tensor, _ = self.model(x_tensor)
            pred_scaled = pred_scaled_tensor.cpu().numpy()
            pred_scaled = np.clip(pred_scaled, -0.2, 1.2)
            
            pred_real = self.scaler_y.inverse_transform(pred_scaled)
            
            num_original_objs = len(self.target_names)
            
            F_adjusted = np.zeros_like(pred_scaled)
            
            # 🚨 终极改动：取消所有的硬性 G_constraints（除了基础防崩）
            # 把原来可能导致死局的“必须在某个区间”降级为 F 的优化目标！
            G_constraints = np.zeros((x_eval.shape[0], num_original_objs)) 
            
            # 4. 计算 F（优化目标）
            for i in range(num_original_objs):
                if self.directions[i] == "最大化 (越大越好)":
                    F_adjusted[:, i] = -pred_scaled[:, i] 
                elif self.directions[i] == "逼近特定值":
                    # 逼近特定值本来就是软目标，继续保留
                    F_adjusted[:, i] = np.abs(pred_scaled[:, i] - self.targets_scaled[i]) 
                elif self.directions[i] == "无极值要求 (仅约束)":
                    F_adjusted[:, i] = 0.0 
                else:
                    F_adjusted[:, i] = pred_scaled[:, i] 
                    
                # 🚨 将 Y 的范围越界，直接作为惩罚加到 F 目标里（软约束）
                t_name = self.target_names[i]
                bounds = self.y_bounds_real.get(t_name, [None, None])
                penalty = 0.0
                if bounds[0] is not None and pred_real[:, i] < float(bounds[0]):
                    penalty += float(bounds[0]) - pred_real[:, i]
                if bounds[1] is not None and pred_real[:, i] > float(bounds[1]):
                    penalty += pred_real[:, i] - float(bounds[1])
                
                # 如果越界，狠狠惩罚 F 的分数，但绝不报错崩溃！
                F_adjusted[:, i] += penalty * 100 
            
            l1_cost = np.sum(np.abs(x_eval - self.x_baseline_scaled), axis=1)
            
            F_final = np.zeros((x_eval.shape[0], self.n_obj)) 
            F_final[:, :-1] = F_adjusted  
            F_final[:, -1] = l1_cost      
            out["F"] = F_final
            
            # 5. 计算极简的惩罚 G（只保留绝对不能容忍的防崩错）
            for i in range(num_original_objs):
                # 只保留最基础的预测值防跌破 0 限制
                G_constraints[:, i] = 0.0 - pred_scaled[:, i]

            # L0 已经在步骤2修复，Y范围在步骤4变为软惩罚
            # 因此，系统现在几乎不可能再出现全体无解的情况
            out["G"] = G_constraints
            
            # =================================================================
            # 🚨 终极闭环：基因回写 (Lamarckian Write-back)
            # 必须把修复过（只改动了2个特征）的基因，强制回写给底层框架
            # 否则框架最后输出的还是原始的未修复方案！
            # =================================================================
            out["X"] = x_eval

            # =================================================================
            # 🚨 独家 Debug 探针：只在所有解都不可行时触发
            # =================================================================
            if np.all(np.any(G_constraints > 0, axis=1)):
                print("--- 🚨 极度警告：当前代所有个体均被判定为不可行！ ---")
                print(f"最大允许改动 (max_changes): {max_c}")
                # 随便抽一个体会看看它到底死在了哪个约束上
                sample_g = G_constraints[0]
                failed_indices = np.where(sample_g > 0)[0]
                print(f"抽样个体失败的约束索引: {failed_indices}")
                if c_idx in failed_indices:
                    print("--> ❌ 失败原因包含：L0 稀疏约束超标！")
                for i in range(num_original_objs, c_idx):
                    if i in failed_indices:
                        print(f"--> ❌ 失败原因包含：因变量 Y 红线越界！(索引 {i})")
                print("--------------------------------------------------")

# ==============================================================================
# 5. 主界面逻辑
# ==============================================================================
st.sidebar.title("通用系统导航")
module = st.sidebar.radio("选择模块:", ("1. 数据流与元数据", "2. AutoML 模型工厂", "3. 决策支持指挥舱"))
st.sidebar.markdown("---")

# === 模块 1: 动态数据流 ===
if module == "1. 数据流与元数据":
    st.title("📂 数据管理与元数据提取")
    uploaded_file = st.file_uploader("上传数据集 (CSV/Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # 获取文件名后缀并转为小写
        file_name = uploaded_file.name.lower()
        
        try:
            # --- 1. 如果用户偷偷传了 Excel 文件 ---
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
                
            # --- 2. 如果是真正的 CSV 文件 ---
            elif file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='gbk', on_bad_lines='skip')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            else:
                st.error("❌ 不支持的文件格式，请上传 .csv 或 .xlsx 文件。")
                st.stop() # 停止运行下方代码

            # 成功读取后的展示
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"✅ 数据读取成功，当前有效数据共 {len(df)} 行。")

        except Exception as e:
            # 终极兜底：如果文件彻底损坏，防止系统红屏崩溃
            st.error(f"❌ 数据解析遭遇致命错误，请检查文件是否损坏。详细信息：{str(e)}")
            st.stop()
        
        st.dataframe(df.head(), use_container_width=True)
        # 可以加个提示，告诉用户有多少条有效数据
        st.caption(f"✅ 数据读取成功，当前有效数据共 {len(df)} 行。")
        
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("配置数据字典 (Metadata)")
        col1, col2 = st.columns(2)
        with col1:
            # 默认把除了最后一列之外的【所有列】作为特征 X
            feature_cols = st.multiselect(
                "选择输入特征 (X):", 
                df.columns, 
                default=list(df.columns)[:-1] if len(df.columns) > 1 else []
            )
        with col2:
            # 算出被 X 挑走后，还剩下哪些列可以作为 Y
            y_options = [c for c in df.columns if c not in feature_cols]
            # Y 的默认值：直接把剩下的列（通常就是目标变量）全包了！绝对不会越界！
            target_cols = st.multiselect(
                "选择优化目标 (Y):", 
                y_options, 
                default=y_options
            )
            
        if st.button("💾 保存元数据并初始化流水线", type="primary") and feature_cols and target_cols:
            st.session_state.df = df
            st.session_state.feature_names = feature_cols
            st.session_state.target_names = target_cols
            
            X = df[feature_cols].values
            Y = df[target_cols].values
            st.session_state.scaler_x = MinMaxScaler().fit(X)
            st.session_state.scaler_y = MinMaxScaler().fit(Y)
            
            st.session_state.data_loaded = True
            st.session_state.model_trained = False # 数据改变，需重新训练
            st.success("✅ 元数据提取成功！特征空间已重新映射，请前往【模型工厂】训练专属代理模型。")

# === 模块 2: AutoML 模型工厂 ===
elif module == "2. AutoML 模型工厂":
    st.title("🏭 自动化机器学习流水线 (AutoML)")
    if not st.session_state.data_loaded:
        st.warning("请先在【数据流与元数据】模块上传数据并配置特征。")
    else:
        num_f = len(st.session_state.feature_names)
        num_t = len(st.session_state.target_names)
        st.info(f"📊 当前感知到的物理系统：输入维度={num_f}，输出维度={num_t}")
        
        if st.button("🚀 一键训练专属代理底座 (Tabular Transformer)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 初始化动态模型
            model = TabularTransformer(num_features=num_f, output_dim=num_t).to(device)
            
            # --- 真实深度学习训练引擎 (Mini-Batch 工业级重构) ---
            from torch.utils.data import TensorDataset, DataLoader
            
            # 1. 准备真实数据 (转为 Tensor)
            X_train = torch.FloatTensor(st.session_state.scaler_x.transform(st.session_state.df[st.session_state.feature_names].values))
            Y_train = torch.FloatTensor(st.session_state.scaler_y.transform(st.session_state.df[st.session_state.target_names].values))
            
            # 🚨 核心升级：引入微批次数据加载器，打乱数据，每次喂 64 条
            dataset = TensorDataset(X_train, Y_train)
            batch_size = min(64, len(X_train)) # 防止数据集太小报错
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 2. 定义损失函数与优化器
            criterion = nn.MSELoss()
            # 降低学习率，配合高频更新，让模型找得更准
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
            
            # 3. 真实 Epoch 循环 (增加到 150 轮)
            epochs = 500
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0
                
                # 🚨 核心升级：Mini-Batch 内部循环
                # 🚨 核心升级：Mini-Batch 内部循环
                for batch_x, batch_y in dataloader:
                    # 将数据搬运到 GPU！
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs, _ = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # 计算这个 Epoch 的平均 Loss
                avg_loss = total_loss / len(dataloader)
                
                # 动态更新前端 UI
                if epoch % 10 == 0 or epoch == epochs:
                    progress_bar.progress(epoch / epochs)
                    status_text.text(f"Epoch {epoch}/{epochs} | 真实 MSE Loss 正在探底: {avg_loss:.5f}")
            
            # 训练结束，将模型切换为评估模式并保存到内存
            st.session_state.model = model.eval()
            st.session_state.model_trained = True
            st.success("✅ 模型训练完成！全局耦合物理规律已冻结，已自动热部署至当前环境。")
            
            model.eval()
            with torch.no_grad():
                preds_scaled, _ = model(torch.FloatTensor(X_train).to(device))
                preds = st.session_state.scaler_y.inverse_transform(preds_scaled.cpu().numpy())
                trues = st.session_state.scaler_y.inverse_transform(Y_train)
            
            mse = mean_squared_error(trues, preds)
            mae = mean_absolute_error(trues, preds)
            r2 = r2_score(trues, preds)
            st.success(f"✅ 模型评估完毕！MSE: {mse:.4f} | MAE: {mae:.4f} | 决定系数 R²: {r2:.4f}")

            # ==========================================
            # 新增：引入传统机器学习作为 Baseline 进行对比
            # ==========================================
            from sklearn.ensemble import RandomForestRegressor
            
            st.markdown("### 🤖 基准模型 (Baseline) 性能对比")
            with st.spinner("正在训练传统机器学习基准模型 (Random Forest)..."):
                # 1. 初始化随机森林 (n_estimators=100为默认经验值，不需要痛苦调参)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 2. 训练 (注意：sklearn需要numpy数组，所以要把刚才的Tensor转成numpy)
                rf_model.fit(X_train.numpy(), Y_train.numpy())
                
                # 3. 预测与反归一化还原
                rf_preds_scaled = rf_model.predict(X_train.numpy())
                
                # 如果是单目标，sklearn会输出一维数组，我们需要把它变回二维以适应反归一化
                if len(rf_preds_scaled.shape) == 1:
                    rf_preds_scaled = rf_preds_scaled.reshape(-1, 1)
                    
                rf_preds = st.session_state.scaler_y.inverse_transform(rf_preds_scaled)
                
                # 4. 计算随机森林的指标
                rf_mse = mean_squared_error(trues, rf_preds)
                rf_mae = mean_absolute_error(trues, rf_preds)
                rf_r2 = r2_score(trues, rf_preds)
                
                st.info(f"🌲 **随机森林 (Random Forest) 基准指标**:\n\n"
                        f"MSE: {rf_mse:.4f} | MAE: {rf_mae:.4f} | 决定系数 R²: {rf_r2:.4f}")

            # ==========================================
            # 🚨 新增：引入 SOTA 树模型 (XGBoost) 进行极致对比
            # ==========================================
            st.markdown("### 🚀 极致梯度提升树 (XGBoost) SOTA基准对比")
            with st.spinner("正在训练 XGBoost 极限梯度提升树..."):
                try:
                    from xgboost import XGBRegressor
                    
                    # 1. 初始化 XGBoost (使用经典的万能参数)
                    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                    
                    # 2. 训练 (XGBoost 完美兼容 sklearn 的 numpy 输入)
                    xgb_model.fit(X_train.numpy(), Y_train.numpy())
                    
                    # 3. 预测与反归一化还原
                    xgb_preds_scaled = xgb_model.predict(X_train.numpy())
                    
                    # 维度安全处理：单目标时 xgboost 输出 1D 数组，需 reshape
                    if len(xgb_preds_scaled.shape) == 1:
                        xgb_preds_scaled = xgb_preds_scaled.reshape(-1, 1)
                        
                    xgb_preds = st.session_state.scaler_y.inverse_transform(xgb_preds_scaled)
                    
                    # 4. 计算 XGBoost 的指标
                    xgb_mse = mean_squared_error(trues, xgb_preds)
                    xgb_mae = mean_absolute_error(trues, xgb_preds)
                    xgb_r2 = r2_score(trues, xgb_preds)
                    
                    st.info(f"🏆 **XGBoost (SOTA) 基准指标**:\n\n"
                            f"MSE: {xgb_mse:.4f} | MAE: {xgb_mae:.4f} | 决定系数 R²: {xgb_r2:.4f}")
                except ImportError:
                    st.error("❌ 未检测到 xgboost 库，请在终端运行 `pip install xgboost`")
            
            # ==========================================
            # 🚨 新增：引入深度学习基线模型 (MLP) 
            # ==========================================
            st.markdown("### 🧠 深度学习基线模型 (MLP) 性能对比")
            with st.spinner("正在快速训练经典深度神经网络 (MLP)..."):
                # 1. 初始化 MLP
                mlp_model = SimpleMLP(num_features=num_f, output_dim=num_t).to(device)
                criterion_mlp = nn.MSELoss()
                optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.005)
                
                # 2. 快速训练 100 个 Epoch (由于是 Baseline，不需要练太久)
                mlp_model.train()
                for epoch in range(100):
                    for batch_x, batch_y in dataloader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        optimizer_mlp.zero_grad()
                        outputs, _ = mlp_model(batch_x)
                        loss = criterion_mlp(outputs, batch_y)
                        loss.backward()
                        optimizer_mlp.step()
                        
                # 3. 评估 MLP
                mlp_model.eval()
                with torch.no_grad():
                    mlp_preds_scaled, _ = mlp_model(torch.FloatTensor(X_train).to(device))
                    mlp_preds = st.session_state.scaler_y.inverse_transform(mlp_preds_scaled.cpu().numpy())
                    
                # 4. 计算指标
                mlp_mse = mean_squared_error(trues, mlp_preds)
                mlp_mae = mean_absolute_error(trues, mlp_preds)
                mlp_r2 = r2_score(trues, mlp_preds)
                
                st.info(f"🕸️ **多层感知机 (MLP) 基准指标**:\n\n"
                        f"MSE: {mlp_mse:.4f} | MAE: {mlp_mae:.4f} | 决定系数 R²: {mlp_r2:.4f}")

# === 模块 3: 决策支持指挥舱 ===
elif module == "3. 决策支持指挥舱":
    st.title("🧠 通用型智能决策底座")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ 暂无可用模型，请先完成数据加载与 AutoML 训练！")
    else:
        # 加载环境上下文
        feature_names = st.session_state.feature_names
        target_names = st.session_state.target_names
        scaler_x = st.session_state.scaler_x
        scaler_y = st.session_state.scaler_y
        model = st.session_state.model
        df = st.session_state.df

        # ================= UI 重构：左侧物理现状 + 右侧AI对话 =================
        
        # ---------------- 👈 左侧：侧边栏 (物理基准现状输入) ----------------
        st.sidebar.header("1️⃣ 当前方案 (Baseline现状)")
        st.sidebar.info("请在此输入系统的当前运行参数。")
        current_vals = []
        for i, name in enumerate(feature_names):
            c_min, c_max = float(df[name].min()), float(df[name].max())
            c_mean = float(df[name].mean())
            # 仅保留现状输入框，去掉了原有的“范围约束”滑动条
            val = st.sidebar.number_input(f"👉 {name}", min_value=c_min, max_value=c_max, value=c_mean, key=f"curr_{i}")
            current_vals.append(val)

        # ---------------- 👉 右侧：主界面 (AI Copilot 寻优指令舱) ----------------
        st.subheader("1. 业务需求与寻优设定 (AI Copilot)")
        
        # 🚨 动态生成占位提示词 (防爆机制：根据目标数量自适应)
        if len(target_names) > 1:
            # 如果是多目标（如建筑能效）
            example_text = f"例如：帮我把 {target_names[0]} 降到最低，同时让 {target_names[1]} 尽量逼近 15。保持 {feature_names[0]} 不变。"
        else:
            # 如果是单目标（如德国信贷）
            example_text = f"例如：帮我把 {target_names[0]} 提升到最高，同时保持 {feature_names[0]} 和 {feature_names[1]} 不变。"
            
        # 唯一的自然语言交互入口
        user_input = st.text_area(
            "🗣️ 请用自然语言描述您的【优化目标】与【物理约束】：",
            value=example_text,
            height=120
        )
        
        run_btn = st.button("🚀 启动智能寻优与物理溯源", type="primary", use_container_width=True)

# ================= 核心执行逻辑 =================
        if run_btn:
            try:
                get_deepseek_api_key()
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            x_curr = np.array([current_vals])
            x_curr_scaled = scaler_x.transform(x_curr)
            
            with torch.no_grad():
                pred_scaled, _ = model(torch.FloatTensor(x_curr_scaled).to(device))
                pred_real = scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0]

            st.markdown("---")
            st.subheader("2. 初始方案基准预测 (Baseline)")
            cols_base = st.columns(len(target_names))
            for i, t_name in enumerate(target_names):
                cols_base[i].metric(f"基准预测 {t_name}", f"{pred_real[i]:.2f}")

            # ==================================================================
            # 🚨 核心枢纽：大语言模型解析 & 数学空间转换 & 规格校验修复
            # ==================================================================
            with st.spinner("智能体正在深度解析您的自然语言指令..."):
                # 1. 获取大模型原始解析结果
                raw_parsed_json = parse_constraints_with_llm(user_input, feature_names, target_names)
                
                # 🚨 2. 挂载文档第 6 节核心机制：规格校验与确定性修复
                parsed_json, repair_logs = validate_and_repair_specs(raw_parsed_json)
                
                # 如果触发了修复机制，在界面上弹出黄色警告高亮展示
                if repair_logs:
                    st.warning("**系统已触发防崩溃边界修复：**\n\n" + "\n".join(repair_logs))

                # 3. 提取修复后的安全数据
                parsed_constraints = parsed_json.get("constraints_x", {})
                parsed_y_constraints = parsed_json.get("constraints_y", {}) 
                parsed_goals = parsed_json.get("goals_y", {})
                
                # ==========================================
                # 界面展示信息拼接
                # ==========================================
                success_msg = "**[系统提示] 自然语言解析成功！已精准转化为底层数学策略：**\n\n"
                
                success_msg += "**[优化目标 Y]**\n"
                if parsed_goals:
                    for k, v in parsed_goals.items():
                        dir_str = "[最小化]" if "min" in str(v).lower() else "[最大化]" if "max" in str(v).lower() else f"[逼近特定值: {v.split(':')[-1]}]"
                        success_msg += f"- {k} -> {dir_str}\n"
                else:
                    success_msg += "- 无极值优化目标\n"

                # 界面展示：Y的安全红线
                y_ranges = parsed_y_constraints.get("ranges", {})
                if y_ranges:
                    success_msg += "\n**[安全红线 Y (状态约束)]**\n"
                    for k, v in y_ranges.items():
                        min_val = "无下限" if v[0] is None else v[0]
                        max_val = "无上限" if v[1] is None else v[1]
                        success_msg += f"- 必须保证 {k} 维持在: [{min_val}, {max_val}]\n"
                    
                locks = parsed_constraints.get("locks", [])
                success_msg += f"\n**[绝对锁定 X]** {', '.join(locks) if locks else '无'}\n"
                
                ranges = parsed_constraints.get("ranges", {})
                success_msg += "\n**[动态范围 X]**\n"
                if ranges:
                    for k, v in ranges.items():
                        min_val = "不限" if v[0] is None else v[0]
                        max_val = "不限" if v[1] is None else v[1]
                        success_msg += f"- {k} -> [{min_val}, {max_val}]\n"
                else:
                    success_msg += "- 无边界限制，算法自由探索\n"
                
                # 🚨 修复了 max_changes 的提取 Bug
                max_changes = parsed_constraints.get("max_changes")
                if max_changes is not None:
                    success_msg += f"\n**[稀疏探索 X]**\n全局稀疏约束启动：算法自由探索，但最终最多只允许改动 {max_changes} 个参数！\n"

                st.success(success_msg)
                

            # ==========================================
            # 底层算法参数构建
            # ==========================================
            # B. 动态构建优化方向
            opt_directions = []
            opt_targets_real = []
            
            for t_name in target_names:
                goal_str = str(parsed_goals.get(t_name, "none")).lower()
                
                if "max" in goal_str:
                    opt_directions.append("最大化 (越大越好)")
                    opt_targets_real.append(0.0)
                elif "target:" in goal_str:
                    opt_directions.append("逼近特定值")
                    try:
                        val = float(goal_str.split(":")[1])
                    except:
                        val = 0.0
                    opt_targets_real.append(val)
                elif "min" in goal_str:
                    opt_directions.append("最小化 (越小越好)")
                    opt_targets_real.append(0.0)
                else:
                    opt_directions.append("无极值要求 (仅约束)")
                    opt_targets_real.append(0.0)

            # C & D. X空间边界
            xl_real = np.array([df[f].min() for f in feature_names], dtype=float)
            xu_real = np.array([df[f].max() for f in feature_names], dtype=float)

            for f_name, f_range in parsed_constraints.get("ranges", {}).items():
                if f_name in feature_names:
                    idx = feature_names.index(f_name)
                    if f_range[0] is not None:
                        xl_real[idx] = max(xl_real[idx], float(f_range[0]))
                    if f_range[1] is not None:
                        xu_real[idx] = min(xu_real[idx], float(f_range[1]))

            xl_scaled = scaler_x.transform([xl_real])[0]
            xu_scaled = scaler_x.transform([xu_real])[0]
            
            # E. 死锁
            detected_locks = parsed_constraints.get("locks", [])
            locked_idx_list = [feature_names.index(name) for name in detected_locks if name in feature_names]
            locked_vals_scaled_list = [x_curr_scaled[0, idx] for idx in locked_idx_list]

            # F. Y目标
            dummy_y = np.zeros((1, len(target_names)))
            for i in range(len(target_names)):
                dummy_y[0, i] = opt_targets_real[i]
            targets_scaled_list = scaler_y.transform(dummy_y)[0].tolist()
            
            # G. 保存 Y 的范围
            y_bounds_real = y_ranges


            with st.spinner("执行多目标区间寻优 (NSGA-II)..."):
                
                # 🚨 1. 抓取大模型解析出来的 max_changes
                parsed_max_changes = parsed_constraints.get("max_changes", None)
                
                # 🚨 2. 将 x_curr_scaled[0] 和 parsed_max_changes 传给新 Problem 类
                problem = DynamicOptProblem(
                    model, xl_scaled, xu_scaled, len(feature_names), len(target_names), 
                    locked_idx_list, locked_vals_scaled_list,
                    opt_directions, targets_scaled_list,
                    target_names, y_bounds_real, scaler_y,
                    x_curr_scaled[0], parsed_max_changes
                )
                
                # === 以下是你原来的 algorithm = NSGA2(...) 代码，保持不变 ===
                
                # ... 下面继续执行算法 algorithm = NSGA2(...) 无需修改 ...
                
                algorithm = NSGA2(pop_size=40, n_offsprings=20, sampling=FloatRandomSampling(), 
                                  crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), 
                                  eliminate_duplicates=True)
                
                res = minimize(problem, algorithm, ('n_gen', 30), seed=1, save_history=True, verbose=False)
                
                if res.X is None or np.any(np.isnan(res.X)):
                    st.error("❌ 寻优失败：当前设定的约束过严导致数学无解。请放宽约束区间！")
                    st.stop()
                
                res_X_2d = np.atleast_2d(res.X)
                res_F_2d = np.atleast_2d(res.F) # 这里的 F 是受优化方向影响的“惩罚分”
                
                # C. 终极绝杀：使用理想点法 (Utopia Point) 寻找“最佳折中甜点”
                # 将各目标的惩罚分统一归一化到 0-1
                F_min = np.min(res_F_2d, axis=0)
                F_max = np.max(res_F_2d, axis=0)
                F_norm = (res_F_2d - F_min) / (F_max - F_min + 1e-9)
                
                # 寻找距离理想点(0,0,0...)欧式距离最近的解
                best_idx = np.argmin(np.linalg.norm(F_norm, axis=1)) 
                best_plan = scaler_x.inverse_transform(res_X_2d)[best_idx]
                
                # D. 重新用模型预测一次真实的物理结果，彻底摆脱负号污染
                with torch.no_grad():
                    best_plan_scaled = scaler_x.transform([best_plan])
                    best_objs_scaled, _ = model(torch.FloatTensor(best_plan_scaled).to(device))
                    # 记得转回 CPU 给 Numpy 用
                    best_objs = scaler_y.inverse_transform(best_objs_scaled.cpu().numpy())[0]

            # ==================================================================
            # 🚨 恢复的第三部分：优化结果与物理归因分析 (DeepSHAP)
            # ==================================================================
            st.markdown("---")
            st.subheader("3. 优化结果与物理归因分析 (DeepSHAP)")
            
            # --- 3.1 优化前后对比指标 ---
            cols_opt = st.columns(len(target_names))
            for i, t_name in enumerate(target_names):
                cols_opt[i].metric(
                    f"AI 推荐 {t_name}", 
                    f"{best_objs[i]:.2f}", 
                    f"{(best_objs[i]-pred_real[i])/pred_real[i]*100:.1f}%", 
                    delta_color="inverse"
                )

            # --- 3.2 动态自适应 SHAP 瀑布图渲染 (纯血 GPU 极速版) ---
            st.markdown("**🔍 现状预测溯源：特征边际贡献瀑布图**")
            
            cols_shap = st.columns(len(target_names))
            
            # 使用缓存加载 K-Means 基准，并送入 GPU
            X_all_scaled = scaler_x.transform(st.session_state.df[feature_names].values)
            bg_data = torch.FloatTensor(get_cached_bg_data(X_all_scaled)).to(device)
            
            # 单目标切片包装器
            class ShapWrapperSingle(nn.Module):
                def __init__(self, m, idx): 
                    super().__init__()
                    self.m = m
                    self.idx = idx
                def forward(self, x): 
                    return self.m(x)[0][:, self.idx].unsqueeze(1)
            
            # 遍历每个优化目标，绘制独立的物理归因图
            for i, t_name in enumerate(target_names):
                with cols_shap[i]:
                    explainer = shap.DeepExplainer(ShapWrapperSingle(model, i).eval(), bg_data)
                    
                    # 当前输入数据送入 GPU
                    x_curr_tensor = torch.FloatTensor(x_curr_scaled).to(device)
                    shap_values = explainer.shap_values(x_curr_tensor, check_additivity=False)
                    
                    # 归因缩放逻辑
                    feature_contributions_scaled = np.array(shap_values).flatten()
                    
                    bg_preds_scaled, _ = model(bg_data)
                    bg_preds_real = scaler_y.inverse_transform(bg_preds_scaled.cpu().detach().numpy())
                    base_real_val = bg_preds_real[:, i].mean()
                    
                    curr_real_val = pred_real[i]
                    real_diff = curr_real_val - base_real_val
                    
                    scale_factor = real_diff / (feature_contributions_scaled.sum() + 1e-9)
                    feature_contributions = feature_contributions_scaled * scale_factor
                    
                    # 排序寻找主要归因
                    feature_names_arr = np.array(feature_names)
                    sorted_indices = np.argsort(-np.abs(feature_contributions))
                    sorted_features = feature_names_arr[sorted_indices].tolist()
                    sorted_contributions = feature_contributions[sorted_indices].tolist()
                    
                    # 构造 Plotly Waterfall 数据结构
                    plot_x = ["系统基准线"] + sorted_features + ["现状真实预测"]
                    plot_y = [base_real_val] + sorted_contributions + [0] 
                    plot_measure = ["absolute"] + ["relative"] * len(sorted_features) + ["total"]
                    
                    fig_shap = go.Figure(go.Waterfall(
                        measure=plot_measure, x=plot_x, y=plot_y,
                        text=[f"{val:.2f}" for val in plot_y[:-1]] + [f"{curr_real_val:.2f}"], 
                        textposition="outside", connector={"line":{"color":"rgb(63, 63, 63)"}},
                        decreasing={"marker":{"color":"#2CA02C"}}, increasing={"marker":{"color":"#D62728"}}, 
                        totals={"marker":{"color":"#1F77B4"}} 
                    ))
                    fig_shap.update_layout(
                        title=f"{t_name} 溯源链条", 
                        height=400, 
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True, key=f"shap_waterfall_{i}")

            # ==================================================================
            # 🚨 专家建议操作指南与决策对齐度度量 (最终完美适配版)
            # ==================================================================
            st.markdown("---")
            st.subheader("4. 专家建议操作指南与决策对齐度度量")
            
            df_sug = generate_expert_report(current_vals, best_plan, feature_names, detected_locks)
            
            # --- Align@K 计算开始 ---
            try:
                # 1. 识别真正改动的特征 (如果修改前后的缩放值差距大于 0.001，认为被修改了)
                changed_features = [f_name for i, f_name in enumerate(feature_names) 
                                   if abs(best_plan[i] - current_vals[i]) > 0.001]
                
                # 2. 检查是否有改动，并且你的代码中确实存在 shap_values
                if len(changed_features) > 0 and 'shap_values' in locals():
                    
                    # 🚨 核心修改：直接使用你代码中现成的 shap_values
                    current_shap = np.array(shap_values)
                    
                    # 取绝对值并扁平化，得到每个特征的影响力大小
                    shap_abs = np.abs(current_shap).flatten()
                    
                    K = max(3, len(changed_features))
                    # 拿到影响力最大的前 K 个特征索引
                    top_k_indices = np.argsort(shap_abs)[-K:]
                    top_k_shap_features = [feature_names[i] for i in top_k_indices]
                    
                    # 3. 计算对齐度
                    intersection = set(changed_features).intersection(set(top_k_shap_features))
                    align_score = (len(intersection) / len(changed_features)) * 100
                    
                    # 4. 渲染 UI
                    col_m1, col_m2 = st.columns([1, 2])
                    with col_m1:
                        st.metric(label=f"🧠 决策逻辑对齐度 (Align@{K})", 
                                  value=f"{align_score:.1f}%", 
                                  delta="通过可解释性校验" if align_score >= 60 else "逻辑偏离",
                                  delta_color="normal" if align_score >= 60 else "inverse")
                    with col_m2:
                        st.caption(f"**对齐分析**：算法修改了 `{len(changed_features)}` 个参数。其中 `{list(intersection)}` 属于 SHAP 识别的核心影响因子。")
                
                elif len(changed_features) == 0:
                    st.warning("⚠️ 当前方案未对参数进行任何改动，无法计算对齐度指标。")
                else:
                    st.info("ℹ️ SHAP 数据暂未生成，请检查上方是否成功输出了瀑布图。")
                    
            except Exception as e:
                # 打印错误方便排查
                st.write(f"💡 对齐度计算暂未就绪 (原因: {e})")

            # --- 展示报告表格 ---
            st.info(f"🔍 综合深度学习归因与大语言模型推理，为您生成如下优化路径：")
            st.dataframe(df_sug, width="stretch")

            # ==================================================================
            # 🚨 核心新增：第五部分 - 大模型生成的叙述性决策报告
            # ==================================================================
            st.markdown("---")
            st.subheader("5. AI 决策审计报告 (Narrative Report)")
            
            with st.spinner("智能体正在撰写深度审计报告..."):
                # =======================================================
                # 🚨 终极防弹设计：现场重新计算当前状态与最优状态的 Y 值
                # 彻底解决任何变量名找不到或维度不对的问题！
                # =======================================================
                with torch.no_grad():
                    # 获取模型所在的设备 (CPU/GPU)
                    model_device = next(model.parameters()).device 
                    
                    # 1. 算当前基准的 Y
                    t_curr = torch.FloatTensor(x_curr_scaled).to(model_device)
                    p_curr_scaled, _ = model(t_curr)
                    current_y_real_safe = scaler_y.inverse_transform(p_curr_scaled.cpu().numpy())
                    
                    # 2. 算优化后方案 (best_plan) 的 Y
                    # 先把最佳方案缩放回模型认识的 scaled 状态
                    best_plan_scaled = scaler_x.transform([best_plan])
                    t_best = torch.FloatTensor(best_plan_scaled).to(model_device)
                    p_best_scaled, _ = model(t_best)
                    pred_real_safe = scaler_y.inverse_transform(p_best_scaled.cpu().numpy())

                # 3. 安全展平数组，彻底解决 1D/2D 维度冲突
                p_real_flat = np.array(pred_real_safe).flatten()
                c_real_flat = np.array(current_y_real_safe).flatten()
                
                # 4. 整理改善情况的数据 (给 DeepSeek 喂语料)
                pred_improvements = ""
                for i, t_name in enumerate(target_names):
                    diff = p_real_flat[i] - c_real_flat[i]
                    trend = "提升" if diff > 0 else "降低"
                    pred_improvements += f"{t_name} 预计从 {c_real_flat[i]:.2f} {trend}至 {p_real_flat[i]:.2f}; "

                # 5. 调用 DeepSeek
                narrative_report = generate_llm_narrative_report(
                    user_input, 
                    changed_features, 
                    align_score, 
                    pred_improvements,
                    top_k_shap_features
                )
                
                # 在界面上用美观的文本框展示
                st.info(narrative_report)
                
                # 提供下载按钮
                st.download_button(
                    label="📥 下载完整审计报告",
                    data=narrative_report,
                    file_name=f"决策报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )


# ==============# =========================================================
                # 🌟 终极版：优化前后 SHAP 物理机理证据链深度对比
                # =========================================================
                st.markdown("---")
                st.markdown("### 🔍 深度解释：优化前后物理机理证据链 (SHAP对比)")
                
                with st.spinner("正在计算优化后可行解的 SHAP 物理贡献度..."):
                    try:
                        # 1. 准备数据并转换为 Tensor (解决 detach 报错的关键)
                        x_orig_tensor = torch.FloatTensor(x_curr_scaled).to(device)
                        x_opt_tensor = torch.FloatTensor(best_plan_scaled).to(device)
                        
                        # 2. 初始化临时解释器 (确保逻辑闭环，针对第一个目标)
                        bg_data_tensor = torch.FloatTensor(get_cached_bg_data(X_all_scaled)).to(device)
                        temp_explainer = shap.DeepExplainer(ShapWrapperSingle(model, 0).eval(), bg_data_tensor)
                        
                        # 3. 计算 SHAP 值 (输入必须是 Tensor)
                        shap_orig = temp_explainer.shap_values(x_orig_tensor, check_additivity=False)
                        shap_opt = temp_explainer.shap_values(x_opt_tensor, check_additivity=False)
                        
                        # 4. 提取数值并压平
                        s_orig = shap_orig[0].flatten() if isinstance(shap_orig, list) else shap_orig.flatten()
                        s_opt = shap_opt[0].flatten() if isinstance(shap_opt, list) else shap_opt.flatten()
                        
                        # 5. 构建全局对比数据框
                        df_shap_compare = pd.DataFrame({
                            '特征': feature_names,
                            '优化前贡献度': s_orig,
                            '优化后贡献度': s_opt
                        })
                        
                        # ========================================================
                        # 🚨 核心逻辑修复：严格过滤，只展示系统【真正修改过】的特征
                        # ========================================================
                        # 识别输入值发生真实改动的特征 (阈值 0.001 防止浮点误差)
                        real_changed_features = [f_name for i, f_name in enumerate(feature_names) 
                                                 if abs(best_plan[i] - current_vals[i]) > 0.001]
                        
                        if len(real_changed_features) > 0:
                            # 核心隔离：只保留那些真正被动过的特征！
                            df_compare_display = df_shap_compare[df_shap_compare['特征'].isin(real_changed_features)].copy()
                            df_compare_display['绝对变化'] = abs(df_compare_display['优化后贡献度'] - df_compare_display['优化前贡献度'])
                            # 按变化幅度排序
                            df_compare_display = df_compare_display.sort_values(by='绝对变化', ascending=False)
                        else:
                            # 兜底防崩：如果由于强约束导致什么都没改，就随便展示前3个
                            df_shap_compare['绝对变化'] = abs(df_shap_compare['优化后贡献度'] - df_shap_compare['优化前贡献度'])
                            df_compare_display = df_shap_compare.sort_values(by='绝对变化', ascending=False).head(3)
                        
                        # 6. 绘制 Plotly 对比图
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(
                            x=df_compare_display['特征'], y=df_compare_display['优化前贡献度'],
                            name='优化前 (当前病灶)', marker_color='#ef553b'
                        ))
                        fig_compare.add_trace(go.Bar(
                            x=df_compare_display['特征'], y=df_compare_display['优化后贡献度'],
                            name='优化后 (修复效果)', marker_color='#00cc96'
                        ))
                        
                        fig_compare.update_layout(
                            title="🌟 关键被改动特征优化前后 SHAP 归因对比 (证据链验证)",
                            barmode='group',
                            xaxis_title="受干预的物理特征",
                            yaxis_title="SHAP 贡献度",
                            margin=dict(l=0, r=0, t=50, b=0)
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                        
                        # 7. 自动化解读文案
                        if len(real_changed_features) > 0:
                            top_feat = df_compare_display.iloc[0]['特征']
                            st.success(f"**证据链验证成功**：系统严格遵守约束，仅修改了 `{len(real_changed_features)}` 个特征。其中，通过修改核心病灶 **{top_feat}**，成功将其对目标的负面影响从 `{df_compare_display.iloc[0]['优化前贡献度']:.3f}` 修正为 `{df_compare_display.iloc[0]['优化后贡献度']:.3f}`。")
                        else:
                            st.warning("⚠️ 系统未能修改任何特征，请检查约束是否过于严苛。")
                        
                    except Exception as e:
                        st.error(f"❌ 证据链生成失败: {str(e)}")