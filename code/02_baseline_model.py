import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # 用來保存模型
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 設置路徑
data_path = '../data/ENB2012_data.xlsx'
model_save_path = '../saved_models/baseline_rf.pkl'
scaler_save_path = '../saved_models/scaler.pkl'

print("正在讀取數據...")
df = pd.read_excel(data_path)

# 重命名列 (保持習慣)
column_mapping = {
    'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area',
    'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation',
    'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Dist',
    'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
}
df = df.rename(columns=column_mapping)

# 2. 數據預處理
# 分離特徵 (X) 和 目標 (Y)
X = df.iloc[:, :8]  # 前8列是輸入
Y = df.iloc[:, 8:]  # 後2列是輸出 (Y1, Y2)

# 歸一化 (Normalization)
# 神經網絡喜歡 0-1 之間的數據，雖然隨機森林不需要，但我們先養成好習慣
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 劃分訓練集和測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

print(f"數據劃分完成：訓練集 {X_train.shape[0]} 條，測試集 {X_test.shape[0]} 條")

# 3. 訓練基準模型 (隨機森林)
# 因為我們要同時預測 Y1 和 Y2，所以用 MultiOutputRegressor
print("正在訓練基準模型 (Random Forest)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

# 4. 預測與評估
print("正在評估模型性能...")
y_pred = model.predict(X_test)

# 計算指標
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"基準模型評估結果：")
print(f"RMSE (均方根誤差): {rmse:.4f} (越小越好)")
print(f"R2 Score (決定係數): {r2:.4f} (越接近1越好)")
print("-" * 30)

# 5. 可視化預測結果 (只畫前50個樣本，不然太亂)
plt.figure(figsize=(12, 5))

# 畫 Y1 (供暖負荷)
plt.subplot(1, 2, 1)
plt.plot(y_test.values[:50, 0], label='True Y1', marker='o')
plt.plot(y_pred[:50, 0], label='Pred Y1', marker='x')
plt.title('Heating Load: True vs Pred')
plt.legend()

# 畫 Y2 (製冷負荷)
plt.subplot(1, 2, 2)
plt.plot(y_test.values[:50, 1], label='True Y2', marker='o', color='orange')
plt.plot(y_pred[:50, 1], label='Pred Y2', marker='x', color='red')
plt.title('Cooling Load: True vs Pred')
plt.legend()

plt.tight_layout()
plt.savefig('../images/baseline_prediction.png')
print("預測對比圖已保存至 images 文件夾")

# 6. 保存模型 (以後可以直接調用，不用重練)
joblib.dump(model, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"模型已保存至: {model_save_path}")