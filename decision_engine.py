import numpy as np
import tensorflow as tf

# ==========================
# Load Trained Model
# ==========================
MODEL_PATH = "multi_output_model_weighted_v3.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")

# ==========================
# Decision Thresholds
# ==========================
EXP_THRESHOLD = 0.35
DIR_THRESHOLD = 0.40

# ==========================
# Decision Rule Parameters
# ==========================
ATR_MIN = 0.0005        # minimum volatility
MAX_SPREAD = 0.0003     # placeholder
MIN_RR = 1.5            # minimum risk reward

# ==========================
# Decision Function
# ==========================
def generate_signal(feature_window, atr, spread, rr_ratio):
    
    # Ensure correct shape
    feature_window = np.expand_dims(feature_window, axis=0)

    # Model prediction
    pred_exp, pred_dir = model.predict(feature_window)

    exp_prob = float(pred_exp[0][0])
    dir_prob = float(pred_dir[0][0])

    print("\nModel Probabilities")
    print("Expansion:", exp_prob)
    print("Direction:", dir_prob)

    # ==========================
    # Rule Filters
    # ==========================

    if exp_prob < EXP_THRESHOLD:
        return "HOLD", exp_prob, dir_prob

    if atr < ATR_MIN:
        return "HOLD", exp_prob, dir_prob

    if spread > MAX_SPREAD:
        return "HOLD", exp_prob, dir_prob

    if rr_ratio < MIN_RR:
        return "HOLD", exp_prob, dir_prob

    # ==========================
    # Direction Decision
    # ==========================

    if dir_prob > DIR_THRESHOLD:
        return "BUY", exp_prob, dir_prob

    elif dir_prob < (1 - DIR_THRESHOLD):
        return "SELL", exp_prob, dir_prob

    else:
        return "HOLD", exp_prob, dir_prob


# ==========================
# Example Test Run
# ==========================
if __name__ == "__main__":

    # Example dummy feature window
    dummy_window = np.random.rand(50, 23)

    atr = 0.001
    spread = 0.0001
    rr_ratio = 2.0

    signal, exp_prob, dir_prob = generate_signal(
        dummy_window,
        atr,
        spread,
        rr_ratio
    )

    print("\nFinal Decision:", signal)
    print("Expansion Probability:", exp_prob)
    print("Direction Probability:", dir_prob)