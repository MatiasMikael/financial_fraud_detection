import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load cleaned data
print("Loading cleaned data...")
data_path = "3_cleaned_data/cleaned_financial_data.csv"
data = pd.read_csv(data_path)

# Sample 10% of the data for faster training
print("Sampling 10% of the data for faster training...")
data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data

# Select features and target
print("Selecting features and target...")
features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
target = "isFraud"
X = data[features]
y = data[target]

# Handle class imbalance using SMOTE
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model without hyperparameter optimization
print("Training the model with Random Forest...")
model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
print("Saving the model...")
model_path = "4_results/simplified_fraud_detection_model.joblib"
joblib.dump(model, model_path)
print(f"Simplified model saved to {model_path}")