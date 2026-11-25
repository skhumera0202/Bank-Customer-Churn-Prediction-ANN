

import joblib
import matplotlib.pyplot as plt

# 1. Load training history
history = joblib.load("training_history.pkl")

# 2. Plot accuracy
plt.figure(figsize=(8, 5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("ANN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_plot.png")
plt.show()

# 3. Plot loss
plt.figure(figsize=(8, 5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("ANN Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_plot.png")
plt.show()

print("Training graphs saved successfully!")
