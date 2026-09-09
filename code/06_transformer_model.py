import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用設備: {device}")

# 2. 準備數據 (和之前一樣)
data_path = '../data/ENB2012_data.xlsx'
df = pd.read_excel(data_path)
column_mapping = {
    'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area',
    'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation',
    'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Dist',
    'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
}
df = df.rename(columns=column_mapping)
feature_names = list(column_mapping.values())[:8]

X = df.iloc[:, :8].values
Y = df.iloc[:, 8:].values

# 歸一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# 轉為 Tensor
X_tensor = torch.FloatTensor(X_scaled).to(device)
Y_tensor = torch.FloatTensor(Y_scaled).to(device)

# 劃分數據
X_train, X_test, y_train, y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# --- 3. 定義 Transformer 模型 (核心升級！) ---
class TabularTransformer(nn.Module):
    def __init__(self, num_features=8, d_model=32, nhead=4, num_layers=2):
        super(TabularTransformer, self).__init__()
        
        # 1. Embedding 層: 把每個特徵的 1 個數值映射成 32 維向量
        # 輸入形狀: (Batch, 8, 1) -> 輸出形狀: (Batch, 8, 32)
        self.feature_embedding = nn.Linear(1, d_model)
        
        # 2. Transformer Encoder 層 (這裡我們手動用 MultiheadAttention 以便提取權重)
        self.d_model = d_model
        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 3. 輸出層
        self.fc = nn.Sequential(
            nn.Flatten(), # 把 (Batch, 8, 32) 展平成 (Batch, 256)
            nn.Linear(num_features * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 預測 Y1, Y2
        )

    def forward(self, x):
        # x 形狀: (Batch, 8) -> 需要擴展成 (Batch, 8, 1) 才能進入 Embedding
        x = x.unsqueeze(-1) 
        
        # 1. Embedding
        x = self.feature_embedding(x) # (Batch, 8, 32)
        
        # 2. Self-Attention (核心！)
        # attn_weights 形狀: (Batch, 8, 8) <- 這就是多變量關聯矩陣！
        attn_output, attn_weights = self.att(x, x, x)
        
        # 殘差連接 + 歸一化
        x = self.norm(x + attn_output)
        
        # Feed Forward Network
        x = x + self.ffn(x)
        
        # 3. 預測輸出
        out = self.fc(x)
        
        return out, attn_weights

# 實例化模型
model = TabularTransformer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. 訓練模型 ---
print("\n開始訓練 Transformer 模型...")
loss_history = []

for epoch in range(300): # 訓練 300 輪
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # 注意：現在 forward 會返回兩個值 (預測值, 權重)
        outputs, _ = model(inputs) 
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    loss_history.append(running_loss / len(train_loader))
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {running_loss:.6f}")

# --- 5. 評估與可視化關聯矩陣 ---
model.eval()
with torch.no_grad():
    y_pred_scaled, attn_weights = model(X_test)
    
    # 計算 R2
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
    y_true = scaler_y.inverse_transform(y_test.cpu().numpy())
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_pred)
    print(f"\nTransformer 模型 R2 Score: {score:.4f}")

    # --- 6. 繪製「多變量關聯熱力圖」(Attention Map) ---
    # 我們取測試集第一個樣本的注意力權重來分析
    # attn_weights 形狀是 (Batch, 8, 8)，我們取平均值看看全局關聯
    avg_attention = attn_weights.mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=feature_names, yticklabels=feature_names, cmap='viridis', annot=False)
    plt.title('Self-Attention Map (Multivariate Correlation)')
    plt.xlabel('Attention Key (Source)')
    plt.ylabel('Attention Query (Target)')
    
    save_path = '../images/transformer_attention.png'
    plt.savefig(save_path)
    print(f"\n[重磅成果] 多變量關聯熱力圖已保存至: {save_path}")
    print("這張圖展示了模型眼中 8 個變量是如何相互『關注』的。")

# 保存模型
torch.save(model.state_dict(), '../saved_models/transformer_model.pth')
print("Transformer 模型已保存。")