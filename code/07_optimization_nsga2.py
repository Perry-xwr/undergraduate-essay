import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. 准备环境 ---
device = torch.device("cpu") # 优化过程不需要GPU，CPU更快

# 重新定义 Transformer 结构 (必须与训练时完全一致)
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
        attn_output, _ = self.att(x, x, x)
        x = self.norm(x + attn_output)
        x = x + self.ffn(x)
        out = self.fc(x)
        return out, _ # 返回预测值和权重

# 加载模型
print("正在加载 Transformer 模型...")
model = TabularTransformer().to(device)
model.load_state_dict(torch.load('../saved_models/transformer_model.pth', map_location=device))
model.eval()

# 重新拟合 Scaler (为了把算法生成的 0-1 数据还原成真实物理量)
data_path = '../data/ENB2012_data.xlsx'
df = pd.read_excel(data_path)
column_mapping = {
    'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area',
    'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation',
    'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Dist',
    'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
}
df = df.rename(columns=column_mapping)
X = df.iloc[:, :8].values
Y = df.iloc[:, 8:].values
scaler_x = MinMaxScaler().fit(X)
scaler_y = MinMaxScaler().fit(Y)

# --- 2. 定义优化问题 (The Optimization Problem) ---
class EnergyOptimizationProblem(Problem):
    def __init__(self):
        # 8个变量，2个目标(Y1, Y2)，0个约束
        # xl (下界) 和 xu (上界) 都设为 0-1 (因为模型输入需要归一化)
        super().__init__(n_var=8, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # x 是遗传算法生成的种群，形状 (Pop_Size, 8)
        
        # 1. 转换为 Tensor
        x_tensor = torch.FloatTensor(x).to(device)
        
        # 2. 使用 Transformer 模型进行预测
        with torch.no_grad():
            # 模型返回 (pred, weights)，我们需要 pred
            y_pred_scaled, _ = model(x_tensor)
            
        # 3. pymoo 默认是"最小化"问题
        # 我们的目标也是最小化能耗 (Y1, Y2)，所以直接返回预测值即可
        # 注意：这里返回的是归一化后的值，为了让算法收敛更快，这没问题
        out["F"] = y_pred_scaled.numpy()

# --- 3. 运行 NSGA-II 算法 ---
print("\n开始执行多目标优化 (NSGA-II)...")
print("目标：同时最小化 [供暖负荷] 和 [制冷负荷]")

problem = EnergyOptimizationProblem()

algorithm = NSGA2(
    pop_size=100,           # 种群大小 (一次试100个方案)
    n_offsprings=50,        # 每次繁衍50个新方案
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 50),          # 迭代 50 代
    seed=1,
    save_history=True,
    verbose=True
)

print(f"\n优化结束！找到 {len(res.X)} 个 Pareto 最优解。")

# --- 4. 结果分析与可视化 ---
# 把最优解 (res.X) 和 对应的能耗 (res.F) 还原回真实数值
opt_params = scaler_x.inverse_transform(res.X)
opt_obj = scaler_y.inverse_transform(res.F)

# 打印前 5 个推荐方案
print("\n=== AI 推荐的 Top 5 节能方案 ===")
results_df = pd.DataFrame(opt_params, columns=list(column_mapping.values())[:8])
results_df['Heating_Load'] = opt_obj[:, 0]
results_df['Cooling_Load'] = opt_obj[:, 1]

# 按供暖负荷排序
print(results_df.sort_values('Heating_Load').head(5))

# 绘制 Pareto 前沿面 (Pareto Front)
plt.figure(figsize=(10, 6))

# 画出原始数据点 (作为背景对比)
original_y = scaler_y.inverse_transform(Y) # 这里可能有维度问题，暂且不画背景，只画最优解
plt.scatter(opt_obj[:, 0], opt_obj[:, 1], c='red', s=50, label='Pareto Optimal Solutions')

plt.title('Multi-Objective Optimization Results (Pareto Front)')
plt.xlabel('Heating Load (kWh)')
plt.ylabel('Cooling Load (kWh)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

save_path = '../images/pareto_front.png'
plt.savefig(save_path)
print(f"\n[重磅成果] Pareto 最优前沿图已保存至: {save_path}")
print("图上的红点，就是系统计算出的『鱼与熊掌兼得』的最佳平衡点。")