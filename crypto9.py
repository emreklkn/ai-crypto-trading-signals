import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Input, Conv1D, MaxPooling1D, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    def __init__(self, symbol="BTCUSDT", interval="1h", seq_len=100, horizon=24):
        self.symbol = symbol
        self.interval = interval
        self.seq_len = seq_len
        self.horizon = horizon
        self.scaler = RobustScaler()
        self.model = None

    def binary_focal_loss(self, gamma=2.5, alpha=0.75):
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = K.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
            return tf.reduce_mean(loss)
        return focal_loss

    def get_binance_data(self):
        limit = 1000
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
        all_data = []

        while True:
            url = (
                f"https://api.binance.com/api/v3/klines?"
                f"symbol={self.symbol}&interval={self.interval}"
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

    def add_features(self, df):
        # Temel özellikler
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Momentum göstergeleri
        df['rsi'] = self.compute_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.compute_macd(df['close'])
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # Trend göstergeleri
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['bb_high'], df['bb_low'] = self.compute_bollinger_bands(df['close'])
        
        # Hacim göstergeleri
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Fiyat değişimleri
        df['price_momentum'] = df['close'] / df['close'].shift(1)
        df['high_low_ratio'] = df['high'] / df['low']
        
        return df

    @staticmethod
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig

    @staticmethod
    def compute_bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        return sma + num_std * std, sma - num_std * std

    def prepare_data(self):
        df = self.get_binance_data()
        df = self.add_features(df)
        df.dropna(inplace=True)

        df['target'] = np.where(df['close'].shift(-self.horizon) > df['close'], 1, 0)
        df.dropna(inplace=True)

        feature_cols = [col for col in df.columns if col != 'target']
        data = df[feature_cols].values
        targets = df['target'].values

        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(targets[i + self.seq_len])
        
        X = np.array(X)
        y = np.array(y)

        # Scale features
        n_features = X.shape[2]
        X_flat = X.reshape(-1, n_features)
        X_flat = self.scaler.fit_transform(X_flat)
        X = X_flat.reshape(X.shape)

        return X, y, df

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            
            # CNN layers
            Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        return model

    def train(self):
        X, y, df = self.prepare_data()
        
        # Train-test split
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Compute class weights
        class_weights = compute_class_weight('balanced',
                                           classes=np.unique(y_train),
                                           y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Build and compile model
        self.model = self.build_model((self.seq_len, X.shape[2]))
        optimizer = Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer,
                         loss=self.binary_focal_loss(),
                         metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # Evaluate
        self.evaluate(X_test, y_test, history, df.index[split_idx+self.seq_len:])
        
        return history, (X_test, y_test)

    def evaluate(self, X_test, y_test, history, test_dates):
        # Model performance
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Predictions
        y_pred = (self.model.predict(X_test) > 0.5).astype(int).flatten()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot training history
        self.plot_training_history(history)
        
        # Plot predictions
        self.plot_predictions(test_dates, y_test, y_pred)

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, dates, y_true, y_pred):
        plt.figure(figsize=(12, 4))
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Direction')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Kullanım
predictor = CryptoPredictor()
history, test_data = predictor.train()
