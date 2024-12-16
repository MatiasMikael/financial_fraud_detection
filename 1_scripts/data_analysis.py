import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned data
print("Loading cleaned data...")
data_path = "3_cleaned_data/cleaned_financial_data.csv"
data = pd.read_csv(data_path)

# Create results folder if not exists
results_path = "4_results"
os.makedirs(results_path, exist_ok=True)

# Helper function for formatting y-axis as millions
def format_millions(value, tick_number):
    return f"{int(value / 1_000_000)}M"

# Analyze transaction types
print("\nAnalyzing transaction types...")
type_counts = data["type"].value_counts()
print(type_counts)
plt.figure(figsize=(8, 6))
sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis")
plt.title("Transaction Types Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_millions))  # Format as millions
plt.savefig(f"{results_path}/transaction_types_distribution.png")
plt.show()

# Analyze fraud by transaction types
print("\nAnalyzing fraud by transaction types...")
fraud_by_type = data[data["isFraud"] == 1]["type"].value_counts()
print(fraud_by_type)

plt.figure(figsize=(8, 6))
sns.barplot(x=fraud_by_type.index, y=fraud_by_type.values, palette="magma")
plt.title("Fraud Distribution by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Count")

# Add exact fraud counts on top of bars
for i, value in enumerate(fraud_by_type.values):
    plt.text(i, value + 50, f"{value}", ha="center", fontsize=10)
    
plt.savefig(f"{results_path}/fraud_by_transaction_type.png")
plt.show()


# Analyze transaction amounts for fraud and non-fraud cases
print("\nAnalyzing transaction amounts for fraud and non-fraud cases...")
plt.figure(figsize=(10, 6))
sns.boxplot(x="isFraud", y="amount", data=data, palette="coolwarm")
plt.title("Transaction Amounts: Fraud vs Non-Fraud")
plt.xlabel("Fraud (0 = No, 1 = Yes)")
plt.ylabel("Transaction Amount")
plt.yscale("log")
plt.savefig(f"{results_path}/transaction_amounts_boxplot.png")
plt.show()

# Analyze balance changes for fraud cases
print("\nAnalyzing balance changes for fraud cases...")
fraud_data = data[data["isFraud"] == 1]
plt.figure(figsize=(12, 6))
sns.scatterplot(x="oldbalanceOrg", y="newbalanceOrig", hue="type", data=fraud_data, palette="tab10")
plt.title("Balance Changes for Fraud Cases")
plt.xlabel("Old Balance (Origin)")
plt.ylabel("New Balance (Origin)")
plt.legend(loc="upper right")
plt.savefig(f"{results_path}/balance_changes_fraud_cases.png")
plt.show()