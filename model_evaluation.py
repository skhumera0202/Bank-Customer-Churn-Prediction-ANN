
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# 1. Load test datasets
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# 2. Load the trained ANN model
model = tf.keras.models.load_model("ann_model.keras")

# 3. Predict churn probabilities
y_prob = model.predict(X_test)

# 4. Convert probabilities into classes
y_pred = (y_prob >= 0.5).astype(int).ravel()

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 6. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Exited"]
    )
)

# 7. ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", roc_auc)
