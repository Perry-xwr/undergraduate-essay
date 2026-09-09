import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 检查是否有 GPU (有显卡就用显卡，没有就用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的计算设备: {device}")

# 2. 准备数据 (和之前一样)
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

# 归一化 (神经网络对数据范围非常敏感，这一步必须做)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler() # 把目标值也缩放到 0-1 之间，训练更容易收敛

X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# 转换为 PyTorch 张量
X_tensor = torch.FloatTensor(X_scaled).to(device)
Y_tensor = torch.FloatTensor(Y_scaled).to(device)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

# 制作数据加載器 (Batch Training)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 定义深度神经网络模型 (DNN)
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        self.model = nn.Sequential(
            # 输入层: 8个特征 -> 64個神經元
            nn.Linear(8, 64),
            nn.ReLU(),           # 激活函數
            nn.Dropout(0.2),     # 防止過擬合
            
            # 隱藏層 1
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 隱藏層 2
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # 輸出層: 輸出 2個值 (Y1, Y2)
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.model(x)

model = EnergyPredictor().to(device)
print(model) # 打印模型結構

# 4. 定義損失函數和優化器
criterion = nn.MSELoss() # 回歸問題通常用均方誤差
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 開始訓練 (Training Loop)
epochs = 500
loss_history = []

print("\n開始訓練神經網絡...")
for epoch in range(epochs):
    model.train() # 切換到訓練模式
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()           # 清空梯度
        outputs = model(inputs)         # 前向傳播
        loss = criterion(outputs, targets) # 計算誤差
        loss.backward()                 # 反向傳播
        optimizer.step()                # 更新權重
        running_loss += loss.item()
    
    # 記錄平均誤差
    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

# 6. 畫出訓練過程 (Loss Curve)
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Deep Learning Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig('../images/dl_training_loss.png')
print("訓練曲線已保存。")

# 7. 模型評估
model.eval() # 切換到評估模式
with torch.no_grad():
    y_pred_scaled = model(X_test)
    
    # 把數據還原回真實數值 (反歸一化)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
    y_true = scaler_y.inverse_transform(y_test.cpu().numpy())
    
    # 計算指標
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

print("-" * 30)
print(f"深度學習模型評估結果：")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)

# 8. 保存 PyTorch 模型
torch.save(model.state_dict(), '../saved_models/dl_model.pth')
print("深度學習模型已保存至 saved_models/dl_model.pth")