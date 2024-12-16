## Financial Fraud Detection Project

This project focuses on building a machine learning pipeline to detect fraudulent financial transactions. The workflow includes data preparation, exploratory analysis, model training, and evaluation to produce a reliable system for identifying suspicious activities.

### **Project Overview**

1. **Goal:**
   - Detect fraudulent transactions with high recall and balanced precision.

2. **Technologies Used:**
   - Python (Pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn).
   - Random Forest for fraud detection.
   - Kaggle API for dataset retrieval.
   - SMOTE for handling class imbalance.

3. **Workflow:**
   - Download data from Kaggle.
   - Clean and preprocess data.
   - Perform exploratory data analysis (EDA).
   - Train and evaluate a Random Forest model.
   - Generate visualizations and metrics for insights.

### **Folder Structure**

- `1_scripts/`: Contains Python scripts for each step in the workflow.
- `2_raw_data/`: Stores the raw dataset downloaded from Kaggle.
- `3_cleaned_data/`: Contains the cleaned dataset.
- `4_results/`: Stores visualizations, trained models, and evaluation results.

**NOTE**: The files 2_raw_data/Synthetic_Financial_datasets_log.csv, 3_cleaned_data/cleaned_financial_data.csv, and 4_results/simplified_fraud_detection_model.joblib were too large to store in this repository due to GitHub's file size limitations.

- **AUC-ROC:** 1.00 (Excellent separation of fraud and non-fraud cases).
- **Confusion Matrix:** High recall for fraud detection but lower precision due to false positives.
- **Insights:** Fraud is most common in "TRANSFER" and "CASH_OUT" transaction types.

### **How to Run the Project**

1. Clone the repository and navigate to the project folder.
2. Install required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute run_all.py to automate the entire workflow:
   ```bash
   python 1_scripts/run_all.py
   ```
4. View results in the `4_results/` folder.

### **Future Work**

- Optimize the model further with hyperparameter tuning.
- Experiment with advanced algorithms (e.g., XGBoost, LightGBM).
- Deploy the model as an API for real-time fraud detection.

### License

Project Code: MIT License.

Dataset: Licensed under CC BY-SA 4.0. https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset

This project demonstrates an end-to-end pipeline for financial fraud detection, combining robust data processing and machine learning techniques.

This project demonstrates an end-to-end pipeline for financial fraud detection, combining robust data processing and machine learning techniques.

