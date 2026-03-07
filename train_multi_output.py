import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D,
    GRU, Dense, Dropout,
    GlobalAveragePooling1D, Concatenate
)
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ==========================
# 1️⃣ Load Data
# ==========================
X = np.load("X.npy")
y_exp = np.load("y_exp.npy")
y_dir = np.load("y_dir.npy")

# ==========================
# 2️⃣ Chronological Split
# ==========================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_exp_train, y_exp_test = y_exp[:split], y_exp[split:]
y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

# ==========================
# 3️⃣ Compute Class Weights
# ==========================
exp_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_exp_train),
    y=y_exp_train
)
dir_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_dir_train),
    y=y_dir_train
)

exp_class_weight = dict(zip(np.unique(y_exp_train), exp_weights))
dir_class_weight = dict(zip(np.unique(y_dir_train), dir_weights))

exp_sample_weights = np.array(
    [exp_class_weight[label] for label in y_exp_train]
)

dir_sample_weights = np.array(
    [dir_class_weight[label] for label in y_dir_train]
)

# ==========================
# 4️⃣ Build Model
# ==========================
input_layer = Input(shape=(X.shape[1], X.shape[2]))

cnn = Conv1D(64, 3, activation='relu')(input_layer)
cnn = MaxPooling1D(2)(cnn)
cnn = GlobalAveragePooling1D()(cnn)

gru = GRU(64)(input_layer)

merged = Concatenate()([cnn, gru])
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.3)(merged)

output_exp = Dense(1, activation='sigmoid', name="expansion")(merged)
output_dir = Dense(1, activation='sigmoid', name="direction")(merged)

model = Model(inputs=input_layer, outputs=[output_exp, output_dir])

model.compile(
    optimizer='adam',
    loss={
        "expansion": "binary_crossentropy",
        "direction": "binary_crossentropy"
    },
    loss_weights={
        "expansion": 1.5,
        "direction": 1.0
    },
    metrics={
        "expansion": ["accuracy"],
        "direction": ["accuracy"]
    }
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    [y_exp_train, y_dir_train],
    validation_data=(X_test, [y_exp_test, y_dir_test]),
    sample_weight=[exp_sample_weights, dir_sample_weights],
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2
)

# ==========================
# Evaluation
# ==========================
pred_exp, pred_dir = model.predict(X_test)

EXP_THRESHOLD = 0.35
DIR_THRESHOLD = 0.40   # 🔥 lowered

exp_pred = (pred_exp > EXP_THRESHOLD).astype(int)
dir_pred = (pred_dir > DIR_THRESHOLD).astype(int)

print("\n=== Expansion Performance ===")
print(classification_report(y_exp_test, exp_pred))

print("\n=== Direction Performance ===")
print(classification_report(y_dir_test, dir_pred))

model.save("multi_output_model_weighted_v3.keras")