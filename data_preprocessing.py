import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# 1. Load dataset
data = pd.read_csv("data/Churn_Modelling.csv")

# 2. Remove unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# 3. Separate independent and dependent variables
X = data.drop("Exited", axis=1)
y = data["Exited"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Label Encoding for Gender
label_encoder = LabelEncoder()

X_train["Gender"] = label_encoder.fit_transform(X_train["Gender"])
X_test["Gender"] = label_encoder.transform(X_test["Gender"])

# 6. One-Hot Encoding for Geography
one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

geo_train = one_hot_encoder.fit_transform(X_train[["Geography"]])
geo_test = one_hot_encoder.transform(X_test[["Geography"]])

geo_columns = one_hot_encoder.get_feature_names_out(["Geography"])

geo_train_df = pd.DataFrame(
    geo_train,
    columns=geo_columns,
    index=X_train.index
)

geo_test_df = pd.DataFrame(
    geo_test,
    columns=geo_columns,
    index=X_test.index
)

X_train = X_train.drop("Geography", axis=1)
X_test = X_test.drop("Geography", axis=1)

X_train = pd.concat([X_train, geo_train_df], axis=1)
X_test = pd.concat([X_test, geo_test_df], axis=1)

# 7. Feature Scaling using StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Display final results
print("\nScaled Training Features:")
print(X_train[:5])

print("\nScaled Testing Features:")
print(X_test[:5])

print("\nFinal Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining Data Mean:", X_train.mean(axis=0).round(2))
print("Training Data Standard Deviation:", X_train.std(axis=0).round(2))
# 9. Save preprocessing objects
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(one_hot_encoder, "one_hot_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nPreprocessing objects saved successfully!")
# 10. Save processed datasets
import numpy as np

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("\nProcessed datasets saved successfully!")