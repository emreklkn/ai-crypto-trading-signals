# -*- coding: utf-8 -*-
"""
Created on Sat May 17 00:56:13 2025

@author: emrek
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1) Teknik indikatörler
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0, 0).rolling(period).mean()
    loss = -delta.where(delta<0, 0).rolling(period).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + num_std*std, sma - num_std*std

# 2) 1 yıllık “paging” ile veriyi çek
def get_binance_1y_ohlcv(symbol="BTCUSDT", interval="1h"):
    limit = 1000
    end_ts = int(datetime.now().timestamp()*1000)
    start_ts = int((datetime.now() - timedelta(days=365)).timestamp()*1000)
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
        last_open = resp[-1][0]
        if last_open >= end_ts:
            break
        start_ts = last_open + 1
        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','num_trades','tb_base_vol','tb_quote_vol','ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df[['open','high','low','close','volume']]

# 3) Veri çek ve indikatörleri ekle
df = get_binance_1y_ohlcv()
df['rsi'] = compute_rsi(df['close'])
df['macd'], df['macd_signal'] = compute_macd(df['close'])
df['bb_high'], df['bb_low'] = compute_bollinger_bands(df['close'])
df.dropna(inplace=True)

# 4) X,y sekanslarını hazırla
seq_len = 50
data = df.values  # numpy array (N x F)
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    # bir sonraki kapanış bir öncekinden yüksekse 1, değilse 0
    y.append(int(data[i+seq_len, 3] > data[i+seq_len-1, 3]))
X = np.array(X)  # (num_samples, seq_len, num_features)
y = np.array(y)

# 5) Train/Test split (chronological, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=False
)

# 6) Scaler’ı yalnızca X_train’e fit et, X_train/X_test’i scale et
n_features = X.shape[2]
scaler = MinMaxScaler()

# reshape for scaler
X_train_flat = X_train.reshape(-1, n_features)
X_test_flat  = X_test.reshape(-1, n_features)

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat  = scaler.transform(X_test_flat)

# geri reshape
X_train = X_train_flat.reshape(X_train.shape)
X_test  = X_test_flat.reshape(X_test.shape)

# 7) Modeli oluştur, compile et
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_len, n_features)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# 8) Callbacks
es  = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

# 9) Eğit
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[es, rlp],
    verbose=2
)

# 10) Test performansı
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 11) Eğitim grafikleri
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title("Accuracy")
plt.tight_layout()
plt.show()

# 12) Son tahminler
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
plt.figure(figsize=(12,3))
plt.plot(df.index[-len(y_test):], y_test, label='Gerçek')
plt.plot(df.index[-len(y_test):], y_pred, label='Tahmin', alpha=0.7)
plt.xticks(rotation=45); plt.legend(); plt.title("Gerçek vs Tahmin")
plt.tight_layout(); plt.show()
