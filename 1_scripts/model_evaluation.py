import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load cleaned data
print("Loading cleaned data...")
data_path = "3_cleaned_data/cleaned_financial_data.csv"
data = pd.read_csv(data_path)

# Select features and target
print("Selecting features and target...")
features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
target = "isFraud"
X = data[features]
y = data[target]

# Load saved model
print("Loading saved model...")
model_path = "4_results/simplified_fraud_detection_model.joblib"  # Updated to simplified model
model = joblib.load(model_path)

# Evaluate on original data (no SMOTE applied)
print("\nEvaluating model on original data...")
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# AUC-ROC score
print("\nCalculating AUC-ROC...")
roc_auc = roc_auc_score(y, y_proba)
print(f"AUC-ROC: {roc_auc:.2f}")

# Confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraud", "Fraud"])
disp.plot()
disp.ax_.set_title("Confusion Matrix (Original Data)")
disp.ax_.set_xlabel("Predicted")
disp.ax_.set_ylabel("Actual")

# Save confusion matrix plot
results_path = "4_results/confusion_matrix_original_data.png"
plt.savefig(results_path)
print(f"Confusion matrix saved to {results_path}")

plt.show()