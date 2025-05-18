import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Teknik indikatörler
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper, lower

# 1 yıllık veriyi "paging" ile çekme
def get_binance_1y_ohlcv(symbol="BTCUSDT", interval="1h"):
    limit = 1000
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    all_data = []

    while True:
        url = (
            f"https://api.binance.com/api/v3/klines?"
            f"symbol={symbol}&interval={interval}"
            f"&startTime={start_ts}&endTime={end_ts}&limit={limit}"
        )
        resp = requests.get(url).json()
        if not resp:
            break
        all_data += resp
        # Son gelen mumun açılış zamanını al, +1 ms ile start_ts güncelle
        last_open = resp[-1][0]
        if last_open >= end_ts:
            break
        start_ts = last_open + 1
        time.sleep(0.1)  # API yavaşlığı için ufak bekleme

    # DataFrame’e dönüştür
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'tb_base_vol', 'tb_quote_vol', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df[['open','high','low','close','volume']]

# Veriyi çek
df = get_binance_1y_ohlcv("BTCUSDT", "1h")

# Teknik indikatörleri hesapla
df['rsi'] = compute_rsi(df['close'])
df['macd'], df['macd_signal'] = compute_macd(df['close'])
df['bb_high'], df['bb_low'] = compute_bollinger_bands(df['close'])
df.dropna(inplace=True)

# Özellikleri ölçeklendir
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Zaman serisi (sekans) ve etiket hazırlığı
X, y = [], []
seq_len = 50
for i in range(len(scaled) - seq_len):
    X.append(scaled[i : i + seq_len])
    # Kapanış fiyatı bir sonraki saate yükseldiyse 1, değilse 0
    y.append(1 if scaled[i+seq_len][3] > scaled[i+seq_len-1][3] else 0)

X = np.array(X)
y = np.array(y)

# LSTM modeli
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitimi 50 epoch’a düşürdük
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)

# Tahmin
y_pred = (model.predict(X) > 0.5).astype(int).flatten()

# Son tahminleri görselleştir
plt.figure(figsize=(12,4))
plt.plot(df.index[-len(y_pred):], y_pred, label='Tahmin (1=Yükseliş)')
plt.title("BTCUSDT 1h – Son Tahminler")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
