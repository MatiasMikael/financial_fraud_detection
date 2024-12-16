## Results Overview
This folder contains the key visualizations and saved models from the financial fraud detection project. Below is a brief description of each file and its purpose.

### Visualizations
1. balance_changes_fraud_cases.png
Scatter plot showing old vs. new balances for fraud cases, highlighting patterns in "TRANSFER" and "CASH_OUT" transactions.

2. confusion_matrix_original_data.png
Confusion matrix showing model performance on original data, useful for evaluating real-world effectiveness.

3. fraud_by_transaction_type.png
Bar chart showing fraud distribution by transaction type, identifying high-risk types like CASH_OUT and TRANSFER.

4. transaction_amounts_boxplot.png
Boxplot comparing transaction amounts for fraud vs. non-fraud cases, showing fraud cases often involve higher amounts.

### Models
1. fraud_detection_model.joblib
Initial Random Forest model, used as a baseline.

2. improved_fraud_detection_model.joblib
Optimized Random Forest model trained with SMOTE for better handling of class imbalance.

3. simplified_fraud_detection_model.joblib
Simplified Random Forest model with fixed parameters, achieving excellent precision, recall, and F1-score, and suitable for production use.

### Summary of Results
The run_all.py script successfully executed the entire workflow, producing the following results:

#### Model Performance
* Precision (Fraud): 0.40
The model identifies 40% of predicted fraud cases correctly.

* Recall (Fraud): 1.00
All actual fraud cases were detected.

* F1-score (Fraud): 0.57
A balance between Precision and Recall for fraud detection.

* AUC-ROC: 1.00
The model perfectly separates fraud and non-fraud cases based on predicted probabilities.

#### Confusion Matrix
The model detects all fraud cases (Recall = 1.00) but produces a high number of false positives, leading to low Precision for fraud cases.

#### Transaction Analysis
* The most frequent transaction types are CASH_OUT (35%) and PAYMENT (34%).
* Fraud is most common in TRANSFER and CASH_OUT transaction types, suggesting these should be prioritized for monitoring.

### Note
Optimization of the model remains incomplete due to hardware limitations, particularly the lack of sufficient computational resources to perform advanced hyperparameter tuning. Further improvements are possible with more powerful hardware.