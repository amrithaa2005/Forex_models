import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================
# 1️⃣ Load EURUSD M15
# ==========================
eur = pd.read_csv("EURUSD_M15.csv")
eur["timestamp"] = pd.to_datetime(eur["timestamp"])
eur = eur.sort_values("timestamp").reset_index(drop=True)

# ==========================
# 2️⃣ Load DXY Daily
# ==========================
dxy = pd.read_csv("DXY_D1.csv")
dxy["date"] = pd.to_datetime(dxy["date"]).dt.tz_localize("UTC")
dxy["dxy_close"] = pd.to_numeric(dxy["dxy_close"], errors="coerce")

# ==========================
# 3️⃣ Engineer DXY Features
# ==========================

dxy["dxy_ema_20"] = dxy["dxy_close"].ewm(span=20).mean()
dxy["dxy_slope"] = dxy["dxy_ema_20"].diff()
dxy["dxy_vol"] = dxy["dxy_close"].pct_change().rolling(20).std()

# USD strength regime
dxy["usd_strength"] = (dxy["dxy_close"] > dxy["dxy_ema_20"]).astype(int)

# Keep only necessary columns
dxy_features = dxy[[
    "date",
    "dxy_close",
    "dxy_ema_20",
    "dxy_slope",
    "dxy_vol",
    "usd_strength"
]]

# ==========================
# 4️⃣ Merge DXY into M15
# ==========================

eur["date"] = eur["timestamp"].dt.floor("D")

df = eur.merge(dxy_features, on="date", how="left")

df = df.dropna().reset_index(drop=True)

# ==========================
# 5️⃣ Existing Advanced EURUSD Features
# ==========================

df["returns"] = df["close"].pct_change()

df["ema_20"] = df["close"].ewm(span=20).mean()
df["ema_50"] = df["close"].ewm(span=50).mean()
df["ema_20_slope"] = df["ema_20"].diff()
df["ema_alignment"] = df["ema_20"] - df["ema_50"]

df["high_low"] = df["high"] - df["low"]
df["atr_14"] = df["high_low"].rolling(14).mean()
df["atr_expansion"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

rolling_mean = df["close"].rolling(20).mean()
rolling_std = df["close"].rolling(20).std()

df["bb_width"] = (rolling_std * 2) / rolling_mean
df["bb_position"] = (df["close"] - rolling_mean) / (2 * rolling_std)

df["rolling_high"] = df["high"].rolling(20).max()
df["rolling_low"] = df["low"].rolling(20).min()

df["breakout_up"] = (df["close"] > df["rolling_high"]).astype(int)
df["breakout_down"] = (df["close"] < df["rolling_low"]).astype(int)

midpoint = (df["rolling_high"] + df["rolling_low"]) / 2
df["range_position"] = (df["close"] - midpoint) / (df["rolling_high"] - df["rolling_low"])

df["vol_compression"] = (df["bb_width"] < df["bb_width"].rolling(50).mean()).astype(int)

df = df.dropna().reset_index(drop=True)

# ==========================
# 6️⃣ Multi-Output Labels
# ==========================

horizon = 4
threshold = 0.0008

df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
df = df.dropna().reset_index(drop=True)

df["y_expansion"] = (abs(df["future_return"]) > threshold).astype(int)
df["y_direction"] = (df["future_return"] > 0).astype(int)

# ==========================
# 7️⃣ Feature Selection
# ==========================

features = [
    # EURUSD
    "open","high","low","close","returns",
    "ema_20","ema_50","ema_20_slope","ema_alignment",
    "atr_14","atr_expansion","rsi",
    "bb_width","bb_position",
    "breakout_up","breakout_down",
    "range_position","vol_compression",

    # DXY Macro Context
    "dxy_close","dxy_ema_20","dxy_slope","dxy_vol","usd_strength"
]

X_raw = df[features]
y_exp = df["y_expansion"]
y_dir = df["y_direction"]

# ==========================
# 8️⃣ Scaling
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ==========================
# 9️⃣ Rolling Sequences
# ==========================

window_size = 50

X_seq = []
y_exp_seq = []
y_dir_seq = []

for i in range(window_size, len(X_scaled)):
    X_seq.append(X_scaled[i-window_size:i])
    y_exp_seq.append(y_exp.iloc[i])
    y_dir_seq.append(y_dir.iloc[i])

X = np.array(X_seq)
y_exp = np.array(y_exp_seq)
y_dir = np.array(y_dir_seq)

print("Final X shape:", X.shape)
print("Expansion distribution:", np.bincount(y_exp))
print("Direction distribution:", np.bincount(y_dir))

np.save("X.npy", X)
np.save("y_exp.npy", y_exp)
np.save("y_dir.npy", y_dir)

print("Macro-aware preprocessing complete.")