import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#LightGBMによる重要度の分析に用いる。
import lightgbm as lgb


#文字列を置き換える及び単位をそろえる。
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

#特徴量の設定:['日付け','終値','始値','高値','安値','出来高','変化率 %']の中から使用する特徴量を選択
#features = ['出来高','終値','変化率 %','安値','range','始値']
features = ['出来高','終値','高値','安値']
target = '次の日の終値'

#入力変数と目的変数をそれぞれ代入する。
X = data_final[features].values
y = data_final[target].values.reshape(-1,1)

"""
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
#"""

#データを分割
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#データを正規化
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step), :]
        Xs.append(v)
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)
#time_step:一度に予測に使われるデータの個数。
time_step = 30
X_train_lstm, y_train_lstm = create_dataset(X_train_scaled, y_train_scaled, time_step)
X_test_lstm, y_test_lstm = create_dataset(X_test_scaled, y_test_scaled, time_step)

#モデルを定義する
n_features = X_train_lstm.shape[2]
model = Sequential()
model.add(Input(shape=(time_step, n_features)))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

n_features = X_train_lstm.shape[2]

#任意の値をそれぞれ指定する。
batch = int(input("バッチサイズを指定してください。:"))
epochs = int(input("エポック数を指定してください。:"))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_lstm, y_train_lstm, batch_size=batch, epochs=epochs, verbose=1)

#予測
train_predict = model.predict(X_train_lstm)
test_predict = model.predict(X_test_lstm)

#元のスケールに戻す
train_predict_orig = scaler_y.inverse_transform(train_predict)
test_predict_orig = scaler_y.inverse_transform(test_predict)

y_train_orig = scaler_y.inverse_transform(y_train_lstm)
y_test_orig = scaler_y.inverse_transform(y_test_lstm)
print("本当の値:",y_test_orig[-1][0])
print("予想値:",test_predict_orig[-1][0])

train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_predict_orig))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_predict_orig))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

#グラフ出力
plt.figure(figsize=(15,6))
plt.plot(y_test_orig, label='Actual Price')
plt.plot(test_predict_orig, label='Predicted Price')
plt.title('stock prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
