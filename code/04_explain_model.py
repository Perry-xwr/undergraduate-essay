import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 检查设备
device = torch.device("cpu") # SHAP 计算在 CPU 上通常更稳定
print("正在进行 SHAP 分析，使用设备: CPU")

# 2. 准备数据
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

# 归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# 转为 Tensor
X_tensor = torch.FloatTensor(X_scaled)
Y_tensor = torch.FloatTensor(Y_scaled) # <--- 补上了这行！

# 划分训练集和测试集
X_train, X_test, _, _ = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

# 3. 重新定义模型结构
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# 4. 加载模型
model = EnergyPredictor().to(device)
model_path = '../saved_models/dl_model.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载成功！")
else:
    print("错误：找不到模型文件，请先运行 03 步！")
    exit()

# ... (前面的代码保持不变，从 "# 5. 开始 SHAP 分析" 这一行开始替换)

# 5. 开始 SHAP 分析
print("\n正在计算 SHAP 值 (这可能需要几秒钟)...")

# 背景样本
background = X_train[:100].to(device)
explainer = shap.DeepExplainer(model, background)

# 计算 SHAP 值
# 注意：有些版本的 SHAP 会返回列表，有些会返回 numpy 数组，下面会自动处理
shap_values_raw = explainer.shap_values(X_test[:200].to(device))

# 自动处理数据格式 (万能补丁)
vals_y1 = None
vals_y2 = None

# 情况 A: 如果返回的是列表 (List)，说明是 [Y1数据, Y2数据]
if isinstance(shap_values_raw, list):
    print("检测到 SHAP 返回格式: List (标准)")
    vals_y1 = shap_values_raw[0]
    vals_y2 = shap_values_raw[1]

# 情况 B: 如果返回的是数组 (Array)，说明是 (样本数, 特征数, 2)
else:
    print(f"检测到 SHAP 返回格式: Array {shap_values_raw.shape} (新版)")
    # 我们需要手动切片：[:, :, 0] 代表取所有样本、所有特征的第1个输出
    vals_y1 = shap_values_raw[:, :, 0]
    vals_y2 = shap_values_raw[:, :, 1]

print("数据解包完成！准备绘图...")

# 特征名称
feature_names = list(column_mapping.values())[:8]
# 准备绘图用的输入数据 (转为 numpy)
X_display = X_test[:200].cpu().numpy()

# 6. 绘制 SHAP 蜂巢图
print("正在生成解释图表...")

# 图1：针对 Y1 (供暖负荷)
try:
    plt.figure()
    plt.title("SHAP Summary for Heating Load (Y1)")
    shap.summary_plot(vals_y1, X_display, feature_names=feature_names, show=False)
    plt.savefig('../images/shap_summary_Y1.png', bbox_inches='tight')
    plt.close()
    print("成功保存: shap_summary_Y1.png")
except Exception as e:
    print(f"绘制 Y1 图表失败: {e}")

# 图2：针对 Y2 (制冷负荷)
try:
    plt.figure()
    plt.title("SHAP Summary for Cooling Load (Y2)")
    shap.summary_plot(vals_y2, X_display, feature_names=feature_names, show=False)
    plt.savefig('../images/shap_summary_Y2.png', bbox_inches='tight')
    plt.close()
    print("成功保存: shap_summary_Y2.png")
except Exception as e:
    print(f"绘制 Y2 图表失败: {e}")

print("\n分析全部完成！")
print("请去 images 文件夹查看生成的解释图。")