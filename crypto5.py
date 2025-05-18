# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:25:22 2025

@author: emrek
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:06:10 2025

@author: emrek
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- Focal Loss tanımı ---
def binary_focal_loss(gamma=2., alpha=.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return focal_loss

# --- Teknik İndikatör Fonksiyonları ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + num_std * std, sma - num_std * std

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def add_features(df):
    df['rsi'] = compute_rsi(df['close'])
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['bb_high'], df['bb_low'] = compute_bollinger_bands(df['close'])
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['atr'] = compute_atr(df)
    df['momentum'] = df['close'].pct_change(periods=10)
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['price_change'] = df['close'].pct_change()
    df['1h_change'] = df['close'].pct_change(periods=1)
    df['4h_change'] = df['close'].pct_change(periods=4)
    df['24h_change'] = df['close'].pct_change(periods=24)
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volume_change'] = df['volume'].pct_change()
    return df

# --- Veri Çekme Fonksiyonu ---
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
        last_open = resp[-1][0]
        if last_open >= end_ts:
            break
        start_ts = last_open + 1
        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'tb_base_vol', 'tb_quote_vol', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- Ana Kod ---

df = get_binance_1y_ohlcv()
df = add_features(df)
df.dropna(inplace=True)

# Hedef değişken (24 saatlik sonrası için getiri var mı yok mu)
horizon = 24
df['target'] = np.where(df['close'].shift(-horizon) > df['close'], 1, 0)
df.dropna(inplace=True)

# Sekans oluşturma
seq_len = 100
data = df.drop('target', axis=1).values
targets = df['target'].values

X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i + seq_len])
    y.append(targets[i + seq_len])
X = np.array(X)
y = np.array(y)

# Train-Test ayırma (son %10 test)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Ölçeklendirme (RobustScaler)
n_features = X.shape[2]
scaler = RobustScaler()

X_train_flat = X_train.reshape(-1, n_features)
X_test_flat = X_test.reshape(-1, n_features)

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

X_train = X_train_flat.reshape(X_train.shape)
X_test = X_test_flat.reshape(X_test.shape)

# Class weight hesapla (dengesiz veri için)
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# --- Model tanımı ---
model = Sequential([
    Input(shape=(seq_len, n_features)),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),              # Önerilen: %20 dropout (eski 0.3'ten azaltıldı)
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),   # Dense katmanları artırıldı
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Derleme
opt = Adam(learning_rate=1e-3)  # Başlangıç lr=0.001
model.compile(optimizer=opt,
              loss=binary_focal_loss(gamma=2., alpha=0.25),
              metrics=['accuracy'])

# Callbacks
es = EarlyStopping(
    monitor='val_loss',
    patience=10,          # Daha sabırlı davranması için 10 yaptım
    restore_best_weights=True,
    min_delta=0.0001,
    verbose=1
)
rlp = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,           # Daha agresif azaltım
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Eğitim
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,        # İstersen 32 deneyebilirsin
    callbacks=[es, rlp],
    class_weight=class_weight_dict,
    shuffle=True,
    verbose=1
)

# Değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Eğitim grafikleri
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.show()

# Tahmin ve görselleştirme
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

plt.figure(figsize=(12, 3))
plt.plot(df.index[-len(y_test):], y_test, label='Gerçek')
plt.plot(df.index[-len(y_test):], y_pred, label='Tahmin', alpha=0.7)
plt.xticks(rotation=45)
plt.legend()
plt.title("Gerçek vs Tahmin")
plt.tight_layout()
plt.show()

# Confusion matrix ve rapor
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
