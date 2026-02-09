# IEEE-CIS Fraud Detection Pipeline 🛡️💳

## 📌 Project Overview
This project focuses on building an end-to-end machine learning pipeline to detect fraudulent online transactions. Using the IEEE-CIS Fraud Detection dataset, the goal was to identify fraudulent activity in a highly imbalanced dataset (~3.5% fraud rate) while maintaining a realistic operational alert rate.

The solution handles data engineering, complex null-value strategies, time-series splitting to prevent data leakage, and utilizes **XGBoost** with **SMOTE** for the final classification.

## 📂 Dataset
* **Source:** IEEE-CIS Fraud Detection (Kaggle)
* **Structure:** Two datasets (`transaction` and `identity`) joined by `TransactionID`.
* **Size:** 590,540 rows × 434 columns (merged).
* **Class Imbalance:**
    * Non-Fraud: 96.5%
    * Fraud: 3.5%

## ⚙️ Methodology & Pipeline

### 1. Data Engineering & Preprocessing
* **Merging:** Left-join identity data onto transactions.
* **Missing Data Strategy:** Treated missingness as information.
    * Categorical nulls filled as "unknown" or "mode".
    * Specific numeric fields (Cards/Addresses) filled with 0 or median.
    * **V-Features:** Dropped sparse V-features (>99% missing) to reduce dimensionality.
* **Outlier Treatment:** Applied IQR-based capping on `TransactionAmt` to reduce the impact of extreme values.
* **Feature Engineering:** Label Encoding for categorical variables.

### 2. Exploratory Data Analysis (EDA) Insights
* **Product Risk:** `ProductCD` "W" has the highest volume of fraud, but "C" has high risk relative to its size.
* **Card Brands:** "Discover" cards showed a higher fraud *rate* compared to Visa/Mastercard, despite lower volume.
* **Email Domains:** `hotmail.com` and `gmail.com` (both sender and recipient) showed elevated fraud rates.
* **Geo-Risk:** Specific address regions (`addr1`, `addr2`) acted as fraud hotspots.

### 3. Modeling Strategy
* **Validation Scheme:** **Time-Series Split** based on `TransactionDT`.
    * Train (80%) -> Validation (20% of Train) -> Test (20% of total).
    * *Why?* To mimic real-world deployment and prevent future data leakage.
* **Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the **sub-training** set (not validation/test) to avoid leakage.
* **Algorithm:** Tuned **XGBoost** Classifier via `RandomizedSearchCV` optimizing for PR-AUC.

## 📊 Model Performance
Evaluated on a hold-out **Test Set** (chronologically the latest 20% of data).

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.8916** |
| **PR-AUC** | **0.5011** |

### Operational Performance (Confusion Matrix)
At a selected threshold optimized for **Recall ≥ 0.80** on the Validation set, the Test set performance was:

| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | 100,843 (TN) | 13,201 (FP) |
| **Actual Positive** | 1,164 (FN) | **2,900 (TP)** |

* **Frauds Caught (Recall):** ~71.4% (2,900 out of 4,064 actual frauds).
* **Alert Rate:** ~13.6% (16,101 transactions flagged for review out of 118,108).
* **Precision:** ~18% (Expected trade-off to maximize recall in fraud detection).

## 🔍 Interpretability (SHAP)
To ensure the model is not a "black box," I used SHAP (SHapley Additive exPlanations) TreeExplainer.
* **Top Drivers:** `V258`, `V317`, `C4`, and `TransactionAmt`.
* **Insights:** High values in specific V-features (rich features representing ranking/counting) strongly pushed the model toward predicting fraud.

## 🛠️ Tools & Technologies
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), SHAP, Matplotlib/Seaborn.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/fraud-detection-pipeline.git](https://github.com/yourusername/fraud-detection-pipeline.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the dataset from Kaggle and place it in the `data/` folder.
4.  Run the Jupyter Notebook `notebooks/fraud_detection_main.ipynb`.

## 📈 Future Improvements
* **Feature Engineering:** Create more time-delta features (e.g., "Time since last transaction for this card").
* **Ensembling:** Stack XGBoost with LightGBM and CatBoost.
* **Threshold Tuning:** Implement dynamic thresholding based on transaction amount tiers.

---
*Created by [Your Name]*
