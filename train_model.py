import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D,
    GRU, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Concatenate
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==========================
# 1️⃣ Load Preprocessed Data
# ==========================
X = np.load("X.npy")
y = np.load("y.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# ==========================
# 2️⃣ Chronological Split (80/20)
# ==========================
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# ==========================
# 3️⃣ Compute Class Weights
# ==========================
class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights_values))

print("Class Weights:", class_weights)

# ==========================
# 4️⃣ Build GRU + CNN Hybrid
# ==========================

input_layer = Input(shape=(X.shape[1], X.shape[2]))

# --- CNN Branch ---
cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = GlobalAveragePooling1D()(cnn)

# --- GRU Branch ---
gru = GRU(64, return_sequences=False)(input_layer)

# --- Merge ---
merged = Concatenate()([cnn, gru])
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.3)(merged)
output = Dense(3, activation='softmax')(merged)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# 5️⃣ Training
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# ==========================
# 6️⃣ Evaluation
# ==========================
loss, accuracy = model.evaluate(X_test, y_test_cat)
print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Trade-only evaluation (exclude HOLD = 1)
trade_indices = y_true != 1
trade_accuracy = np.mean(y_pred[trade_indices] == y_true[trade_indices])

print("\nTrade-Only Accuracy (BUY/SELL only):", trade_accuracy)

# ==========================
# 7️⃣ Save Model
# ==========================
model.save("gru_cnn_model.keras")

print("\nModel training complete and saved.")