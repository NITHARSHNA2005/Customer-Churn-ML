# Bank Customer Churn Prediction

This project focuses on predicting customer churn—a critical challenge for any subscription or banking-based business. Identifying customers who might leave before they actually do is the key to proactive retention and saving revenue. 

I took a **hybrid approach** here. The goal wasn't just to throw a complex machine learning model at a dataset and call it a day. Instead, I combined the raw predictive power of ML (specifically Logistic Regression and XGBoost) with a straightforward, rule-based risk scoring system. This ensures the output is not just highly accurate, but also easily understandable and actionable for business teams.

## Objectives

- **Predict Churn:** Build models to identify customers likely to leave.  
- **Handle Imbalance:** Manage imbalanced data without changing the original dataset.  
- **Improve Predictions:** Tune the threshold to increase recall while keeping precision balanced.  
- **Compare Models:** Compare a simple model with a more advanced model.  
- **Explain Results:** Use rule-based logic to understand why a customer is at risk.  

---

##  The Data Landscape

I used the **Churn Modelling Dataset** from Kaggle, consisting of 10,000 customer records and 14 features. 

**Target Variable:** `Exited` (0 = Stayed, 1 = Churned)

To make the data more intuitive for business logic, I mapped several technical columns to standard terms:
- `EstimatedSalary` → `Income`
- `NumOfProducts` → `Purchases`
- `IsActiveMember` → `Membership`
- `Exited` → `Churn`

### Preprocessing & Feature Engineering

Before training any models, I thoroughly cleaned and prepped the data:
- Dropped irrelevant identifiers (`RowNumber`, `CustomerId`, `Surname`) because they don't contribute to behavioral patterns.
- Handled categorical data using Label Encoding (`Gender`) and One-Hot Encoding (`Geography`).
- Applied `StandardScaler` to normalize features (especially crucial for Logistic Regression).
- Split the dataset into 80% training / 20% testing using **stratified sampling** to preserve the class balance.

I also engineered a few custom features to help the models pick up on the nuances of customer behavior:
- **`BalanceSalaryRatio`:** (Balance / Salary) Let's us know how much of their income sits in the bank.
- **`IsHighRiskProfile`:** Flags users over 45 who are inactive—my early analysis showed this specific segment had an unusually high churn rate.
- **`ProductsPerTenure`:** Highlights how actively a customer uses services throughout their relationship.
- **`AgeBalanceProduct`:** Combines age with active balance usage.

---

## Handling the Imbalanced Data

Right out of the gate, the dataset is imbalanced (~80% stay vs. ~20% churn). If left unchecked, models simply default to predicting the majority class.

Instead of inserting synthetic data (like SMOTE) or duplicating records—which can lead to overfitting and loss of real-world authenticity—I went with an industry-standard best practice: **Class Weighting**. 
- For Logistic Regression, I utilized `class_weight='balanced'`.
- For XGBoost, I brought in `scale_pos_weight`.

This allowed me to preserve the real-world dataset directly while forcing the models to pay much closer attention to the minority class (the churners).

---

## The Models

### 1. The Baseline: Logistic Regression
I implemented Logistic Regression to set a baseline on performance.
- **Accuracy:** 74.2%
- **Recall:** 63.4%
- **Precision:** 41.3%
- **ROC-AUC:** 77.9%

*Takeaway:* It performed decently but struggled to capture complex, non-linear relationships in the customer data. It missed too many churn risks.

### 2. The Heavy Hitter: XGBoost
I brought in XGBoost as the primary model due to its fantastic ability to handle non-linear relationships and structured tabular data. Even before threshold tuning (at 0.5), it outperformed the baseline considerably:
- **Accuracy:** 80.0%
- **Recall:** 73.7%
- **ROC-AUC:** 86.6%

---

## Threshold Tuning: Finding the Sweet Spot

Using a default probability threshold of 0.5 isn't optimal for imbalanced classification scenarios. I systematically tested thresholds ranging from 0.20 to 0.60 to find the sweet spot that maximized **Recall** (catching more churning customers) while ensuring **Precision** didn't drop below an acceptable standard (≥ 0.45).

**Optimal Threshold Found: 0.42**
- **Recall Surged To:** 80.6% (Up from 73.7%)
- **Precision:** 45.2%
- **F1 Score:** 58.0%
- **Accuracy:** 76.2%

While there was a slight dip in overall accuracy, the massive improvement in recall (successfully identifying 80% of churners instead of ~73%) makes it an incredibly worthwhile trade-off for any retention team.

---

##  What's Actually Driving Churn? (Feature Insights)

When I analyzed the feature importance, XGBoost relied heavily on:
1. `IsHighRiskProfile` 
2. `NumOfProducts`
3. `Age`
4. `IsActiveMember`

This aligns perfectly with domain knowledge: customer age demographics, activity levels, and broader behavioral patterns are massive indicators of loyalty. 

---

## Rule-Based Risk Scoring

Machine learning is powerful, but business teams need clear, interpretable reasons for why someone is considered a risk. To accompany the ML model, I designed a separate, interpretable rule-based scoring system derived from those feature insights. Each risk factor adds to a cumulative score:

| Score | Risk Level | Number of Customers | Actual Churn Rate |
| :--- | :--- | :--- | :--- |
| **0–2** | Low Risk | 4,076 | 10.6% |
| **3–5** | Medium Risk | 5,156 | 20.5% |
| **≥ 6** | High Risk | 768 | **71.1%** |

*Insight:* The strong separation in the actual churn rate validates this rule effectiveness. High-risk customers churned at 71.1%, validating both the machine learning patterns and the business logic!

---

## Final Conclusion

This project successfully proves that combining predictive ML models with simpler rule-based logic is incredibly effective. 

1. **XGBoost** (with a carefully tuned threshold of 0.42) proved to be vastly superior to Logistic Regression in detecting churn.
2. Handling imbalanced data successfully through parameter weighting maintained the integrity of the original dataset.
3. The custom-engineered features effectively captured what raw data couldn't.
4. The integrated rule-based logic bridges the gap between complex algorithms and clear, actionable insights for the business.
