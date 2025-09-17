import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import japanize_matplotlib


def normalize(values):
    if 'M' in values:
        return float(values.replace('M','')) * 1000000
    elif 'B' in values:
        return float(values.replace('B','')) * 1000000000
    elif '%' in values:
        return float(values.replace('%',''))

#ファイルのパスを指定する。
path = input("CSVファイルのパスを入力してください。:")
data = pd.read_csv(path)

desinate_1 = ['日付け','終値','始値','高値','安値','出来高','変化率 %']
desinate_2 = ['終値','始値','高値','安値','出来高']
data_1 = data[desinate_1]
data_1['出来高'] = data['出来高'].apply(normalize)
data_1['変化率 %'] = data['変化率 %'].apply(normalize)
data_1['日付け'] = pd.to_datetime(data_1['日付け'])

data_1 = data_1.sort_values('日付け',ascending=True).reset_index(drop=True)
data_1['次の日の終値'] = data_1['終値'].shift(-1)

print(data_1.describe())

data_1['range'] = data_1['高値'] - data_1['安値']
data_1['sf_range'] = data_1['終値'] - data_1['始値']

data_final = data_1.dropna()
#相関係数を計算
matrix = data_1.corr()
#ヒートマップを出力
plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('特徴量の相関ヒートマップ')
plt.show()

"""
#特徴量の設定:['日付け','終値','始値','高値','安値','出来高','変化率 %']の中から使用する特徴量を選択
#features = ['出来高','終値','変化率 %','安値','range','始値']
features = ['出来高','終値','高値','安値']
target = '次の日の終値'

#入力変数と目的変数をそれぞれ代入する。
X = data_final[features].values
y = data_final[target].values.reshape(-1,1)


##LightBGMによる特徴量の重要度を測る際に使用可能
lgb_X = pd.DataFrame(X,columns=features)
lgb_y = pd.DataFrame(y)

model = lgb.LGBMRegressor(random_state=42)
model.fit(lgb_X,lgb_y)

importance = model.feature_importances_
feature_name = lgb_X.columns


feature_importance_df = pd.DataFrame({
    'feature': feature_name, 
    'importance': importance
}).sort_values('importance', ascending=False)

print("---特徴量の重要度---")
print(feature_importance_df)

plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['feature'],feature_importance_df['importance'])
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.title('LightGBMによる特徴量の重要度')
plt.gca().invert_yaxis()
plt.show()
"""