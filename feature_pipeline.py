import pandas as pd
import numpy as np


WINDOW_SIZE = 50
FEATURE_COUNT = 23


def generate_features(df):

    # Basic indicators
    df["returns"] = df["close"].pct_change()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["returns"].rolling(10).std()

    df = df.fillna(0)

    # Convert to numpy
    features = df.values

    # Ensure exactly 23 features
    if features.shape[1] < 23:
        padding = np.zeros((features.shape[0], 23 - features.shape[1]))
        features = np.concatenate([features, padding], axis=1)

    elif features.shape[1] > 23:
        features = features[:, :23]

    return features


def build_feature_window(features):

    if len(features) < WINDOW_SIZE:
        raise ValueError("Not enough data for window")

    window = features[-WINDOW_SIZE:]

    # Ensure correct shape
    window = window[:, :FEATURE_COUNT]

    return window


if __name__ == "__main__":

    # Example dataset
    # Generate dummy market data
    data = {
    "open": np.random.rand(100),
    "high": np.random.rand(100),
    "low": np.random.rand(100),
    "close": np.random.rand(100),
    "volume": np.random.rand(100)
    }

    df = pd.DataFrame(data)

    features = generate_features(df)

    window = build_feature_window(features)

    print("Feature window shape:", window.shape)