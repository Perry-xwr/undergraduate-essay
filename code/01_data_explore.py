import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 設置文件路徑
# 使用相對路徑：從 code 文件夾往上一級走，進入 data 文件夾
data_path = '../data/ENB2012_data.xlsx'
save_dir = '../images/'

print(f"正在嘗試讀取數據：{os.path.abspath(data_path)} ...")

try:
    # 2. 讀取 Excel 數據
    df = pd.read_excel(data_path)
    
    # 3. 重命名列 (根據 UCI 官方文檔定義)
    # X1-X8 是房屋設計參數，Y1-Y2 是我們預測的能耗目標
    column_mapping = {
        'X1': 'Relative_Compactness',   # 相對緊湊度
        'X2': 'Surface_Area',           # 表面積
        'X3': 'Wall_Area',              # 牆面積
        'X4': 'Roof_Area',              # 屋頂面積
        'X5': 'Overall_Height',         # 總高度
        'X6': 'Orientation',            # 朝向
        'X7': 'Glazing_Area',           # 玻璃面積
        'X8': 'Glazing_Area_Dist',      # 玻璃分布區域
        'Y1': 'Heating_Load',           # 供暖負荷 (目標1)
        'Y2': 'Cooling_Load'            # 製冷負荷 (目標2)
    }
    df = df.rename(columns=column_mapping)
    
    print("數據讀取成功！")
    print(f"數據規模：{df.shape} (768行樣本, 10個變量)")
    print("\n前5行數據預覽：")
    print(df.head())

    # 4. 繪製多變量關聯熱力圖 (Multivariate Correlation)
    # 這是你畢設題目中「關聯預測」的核心依據
    plt.figure(figsize=(10, 8))
    # 計算相關係數矩陣
    corr = df.corr()
    # 畫圖
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature-Target Correlation Analysis')
    
    # 保存圖片
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path)
    print(f"\n[成功] 關聯熱力圖已保存至：{os.path.abspath(save_path)}")
    
    print("\n恭喜！第一步數據探索完成。")
    print("請去 images 文件夾查看生成的圖片。")

except FileNotFoundError:
    print("\n[錯誤] 找不到數據文件！")
    print("請確認 ENB2012_data.xlsx 是否已經放在 data 文件夾中。")
except Exception as e:
    print(f"\n[錯誤] 發生了預料之外的問題：{e}")