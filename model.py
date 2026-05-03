import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate simple regression data
x = np.linspace(-1, 1, 200).reshape(-1, 1)
noise = np.random.normal(0, 0.15, x.shape)
y = 3 * x + 2 + noise

# Train/test split
split_index = int(0.8 * len(x))
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=40,
    validation_split=0.2,
    verbose=0
)

# Evaluate model
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

# Predict results
y_pred = model.predict(x_test, verbose=0)

# Save graph result
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, label="Actual Data")
plt.plot(x_test, y_pred, label="Model Prediction")
plt.title("Simple Regression Model Results")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_results.png")
plt.close()

# Save metrics
with open("metrics.txt", "w") as file:
    file.write("Model Training Results\n")
    file.write("======================\n")
    file.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
    file.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test MAE: {test_mae:.4f}\n")

print("Training complete.")
print("Generated files: model_results.png and metrics.txt")
