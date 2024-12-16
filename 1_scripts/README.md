## Scripts Overview

This folder contains the Python scripts used in the financial fraud detection project. Below is a brief description of each script and its purpose:

### **1. data_download.py**
- **Purpose:** Downloads the dataset from Kaggle using the Kaggle API and saves it to the `2_raw_data` folder.
- **Output:** Raw data files.

### **2. data_cleaning.py**
- **Purpose:** Cleans the raw dataset by removing invalid or missing values.
- **Output:** Cleaned dataset saved to the `3_cleaned_data` folder.

### **3. data_analysis.py**
- **Purpose:** Generates visualizations and insights from the dataset, such as transaction distributions and fraud analysis.
- **Output:** Visualizations saved to the `4_results` folder.

### **4. model_training.py**
- **Purpose:** Trains a Random Forest model using the cleaned dataset. Includes steps like feature selection, data splitting, and applying SMOTE for class imbalance.
- **Output:** Trained model saved as `fraud_detection_model.joblib` or `simplified_fraud_detection_model.joblib` in the `4_results` folder.

### **5. model_evaluation.py**
- **Purpose:** Evaluates the trained model on the original dataset. Calculates metrics like AUC-ROC and generates a confusion matrix.
- **Output:** Confusion matrix saved to the `4_results` folder.

### **6. run_all.py**
- **Purpose:** Automates the entire workflow, including data download, cleaning, analysis, model training, and evaluation.
- **Output:** All results (visualizations, models, metrics) are saved in their respective folders.

---

### Usage Instructions

1. Ensure all dependencies are installed (refer to `requirements.txt`).
2. Run individual scripts for specific tasks or execute `run_all.py` to automate the entire workflow.

### Folder Structure

- **`2_raw_data/`**: Raw dataset files.
- **`3_cleaned_data/`**: Cleaned dataset.
- **`4_results/`**: Visualizations, trained models, and evaluation results.

This structure ensures clarity and separation of tasks throughout the project.