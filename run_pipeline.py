import pandas as pd
import numpy as np
import datetime

from feature_pipeline import generate_features, build_feature_window
from decision_engine import generate_signal


# ==========================
# Load Market Data
# ==========================

data = {
    "open": np.random.rand(100),
    "high": np.random.rand(100),
    "low": np.random.rand(100),
    "close": np.random.rand(100),
    "volume": np.random.rand(100)
}

df = pd.DataFrame(data)


# ==========================
# Feature Generation
# ==========================

features = generate_features(df)

window = build_feature_window(features)

print("Feature window shape:", window.shape)


# ==========================
# Market Conditions
# ==========================

atr = 0.001
spread = 0.0001
rr_ratio = 2.0


# ==========================
# Generate Signal
# ==========================

signal, exp_prob, dir_prob = generate_signal(
    window,
    atr,
    spread,
    rr_ratio
)

print("\n===== SIGNAL OUTPUT =====")
print("Signal:", signal)
print("Expansion Probability:", exp_prob)
print("Direction Probability:", dir_prob)


# ==========================
# Decision Logging
# ==========================

log_data = {
    "timestamp": datetime.datetime.now(),
    "signal": signal,
    "expansion_probability": exp_prob,
    "direction_probability": dir_prob,
    "atr": atr,
    "spread": spread,
    "rr_ratio": rr_ratio
}

log_df = pd.DataFrame([log_data])

log_df.to_csv(
    "decision_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("decision_log.csv"),
    index=False
)

print("\nDecision logged successfully.")