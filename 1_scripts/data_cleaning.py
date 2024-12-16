import pandas as pd

# Load the dataset
print("Loading the dataset...")
data_path = "2_raw_data/Synthetic_Financial_datasets_log.csv"
data = pd.read_csv(data_path)

# Display basic information about the dataset
print("\nDataset Info:")
print(data.info())

# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values...")
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Analyze the number of fraudulent transactions
print("\nAnalyzing fraudulent transactions...")
fraud_count = data["isFraud"].sum()
total_count = len(data)
fraud_percentage = (fraud_count / total_count) * 100
print(f"Total transactions: {total_count}")
print(f"Fraudulent transactions: {fraud_count} ({fraud_percentage:.2f}%)")

# Remove rows with negative balances
print("\nRemoving rows with negative balance values...")
cleaned_data = data[
    (data["oldbalanceOrg"] >= 0) &
    (data["newbalanceOrig"] >= 0) &
    (data["oldbalanceDest"] >= 0) &
    (data["newbalanceDest"] >= 0)
]
print(f"Rows after cleaning: {len(cleaned_data)} (removed {len(data) - len(cleaned_data)})")

# Save the cleaned data to a new CSV file
print("\nSaving the cleaned data...")
output_path = "3_cleaned_data/cleaned_financial_data.csv"
cleaned_data.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")