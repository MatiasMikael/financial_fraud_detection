import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Paths
data_path = "3_cleaned_data/cleaned_financial_data.csv"
model_path = "4_results/simplified_fraud_detection_model.joblib"
results_folder = "4_results"

# Step 1: Data Cleaning and Loading
print("Step 1: Loading cleaned data...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cleaned data file not found at {data_path}")

data = pd.read_csv(data_path)
print(f"Loaded data with {len(data)} rows.")

# Step 2: Feature Selection
print("Step 2: Selecting features and target...")
features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
target = "isFraud"
X = data[features]
y = data[target]

# Step 3: Model Training
print("Step 3: Training the model...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, model_path)
print(f"Model trained and saved to {model_path}")

# Step 4: Model Evaluation
print("Step 4: Evaluating the model...")
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

print("\nClassification Report:")
print(classification_report(y, y_pred))

print("\nCalculating AUC-ROC...")
roc_auc = roc_auc_score(y, y_proba)
print(f"AUC-ROC: {roc_auc:.2f}")

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraud", "Fraud"])
disp.plot()
disp.ax_.set_title("Confusion Matrix (Original Data)")
disp.ax_.set_xlabel("Predicted")
disp.ax_.set_ylabel("Actual")

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

confusion_matrix_path = os.path.join(results_folder, "confusion_matrix_original_data.png")
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to {confusion_matrix_path}")
plt.show()

# Step 5: Generate Insights (Optional)
print("Step 5: Generating insights...")
transaction_type_counts = data["type"].value_counts()
print(f"\nTransaction type counts:\n{transaction_type_counts}")