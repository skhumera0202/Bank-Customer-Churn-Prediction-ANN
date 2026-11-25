
# Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## Project Overview

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer is likely to leave the bank or continue using its services.

The goal is to build a machine learning classification model that can help identify customers who may be at risk of leaving.

The project covers the complete machine learning workflow, from data exploration and preprocessing to model training, evaluation, prediction, and a Streamlit web application.

## Problem Statement

Customer churn is an important challenge for banks. Identifying customers who are likely to leave can help businesses understand customer behavior and improve retention strategies.

This project uses customer information such as credit score, age, geography, balance, and account activity to estimate the likelihood of churn.

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Dataset

The project uses the Bank Customer Churn Modelling dataset.

- Total records: 10,000
- Total original columns: 14
- Target column: `Exited`
- `0`: Customer stays
- `1`: Customer leaves

Dataset source:

https://github.com/pdt0208/Banking-Churn-Modelling/blob/main/Churn_Modelling.csv

## Project Workflow

1. Data Exploration
2. Data Preprocessing
3. Feature Encoding
4. Feature Scaling
5. ANN Model Building
6. Model Training
7. Model Evaluation
8. Prediction System
9. Streamlit Web Application

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns: `RowNumber`, `CustomerId`, and `Surname`.
- Encoded the Gender column using Label Encoding.
- Applied One-Hot Encoding to the Geography column.
- Split the dataset into training and testing sets (80:20).
- Applied StandardScaler to scale numerical features.
- Saved the encoders and scaler for future predictions.

After preprocessing:

- Training samples: 8,000
- Testing samples: 2,000
- Input features: 12

## ANN Model Architecture

The ANN model was built using TensorFlow and Keras.

| Layer | Neurons | Activation |
|---|---:|---|
| Input Layer | 12 features | - |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output Layer | 1 | Sigmoid |

### Model Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Batch Size: 32
- Maximum Epochs: 50
- Early Stopping: Enabled
- Class Weighting: Applied to address class imbalance

## Model Evaluation

The model was evaluated on the test dataset containing 2,000 records.

### Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 79.95% |
| ROC-AUC | 0.8580 |
| Churn Precision | 0.50 |
| Churn Recall | 0.75 |
| Churn F1-Score | 0.60 |

### Confusion Matrix

| | Predicted Stay | Predicted Churn |
|---|---:|---:|
| Actual Stay | 1292 | 301 |
| Actual Churn | 100 | 307 |

The model identifies approximately 75% of the actual churn cases in the test dataset.

Class weighting helped improve churn detection, although it also increased false-positive predictions.

### Training Visualizations

#### Model Accuracy

![Model Accuracy](accuracy_plot.png)

#### Model Loss

![Model Loss](loss_plot.png)

## Prediction System

A separate prediction module was developed to make predictions using the trained ANN model.

The prediction system:

- Loads the trained ANN model.
- Loads the saved encoders and scaler.
- Accepts customer details as input.
- Applies the same preprocessing steps used during training.
- Predicts the probability of customer churn.

## Streamlit Web Application

A Streamlit application was created to allow users to enter customer details and view the model's churn prediction.

### Run the Application Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Bank-Customer-Churn-Prediction-ANN.git
```

2. Navigate to the project directory:

```bash
cd Bank-Customer-Churn-Prediction-ANN
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## Project Structure

```text
Bank-Customer-Churn-Prediction-ANN/
│
├── app.py
├── data_exploration.py
├── data_preprocessing.py
├── model_training.py
├── model_evaluation.py
├── model_visualization.py
├── prediction.py
│
├── ann_model.keras
├── label_encoder.pkl
├── one_hot_encoder.pkl
├── scaler.pkl
│
├── accuracy_plot.png
├── loss_plot.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

The original dataset and processed training arrays are excluded from version control.

## Key Learnings

Through this project, I gained practical experience with:

- Building an ANN classification model.
- Performing data preprocessing and feature encoding.
- Applying feature scaling.
- Understanding class imbalance and class weighting.
- Training and evaluating neural networks.
- Using Early Stopping to help control overfitting.
- Saving and loading trained models.
- Building an interactive application using Streamlit.

## Future Improvements

- Experiment with different hidden layers and neuron configurations.
- Perform hyperparameter tuning.
- Improve churn prediction performance.
- Deploy the Streamlit application online.

## Disclaimer

The predictions are model estimates based on the input customer information. They are not guarantees of actual customer behavior.

## Author

**Humera Shaikh**

GitHub: https://github.com/skhumera0202
