
import pandas as pd

# Load dataset
data = pd.read_csv("data/Churn_Modelling.csv")

# Display first five rows
print("First five rows:")
print(data.head())

# Dataset shape
print("\nDataset Shape:")
print(data.shape)

# Column names
print("\nColumn Names:")
print(data.columns.tolist())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())
