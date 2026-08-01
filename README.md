# 📊 Customer Intelligence System

An end-to-end customer analytics platform built on the Telco Customer Churn dataset — combining churn prediction, customer segmentation, tenure forecasting, and anomaly detection into a single interactive Streamlit application, backed by a MySQL data pipeline.

## 🚀 Live Demo

**[Try the app here](https://customer-intelligence-system-hxhnhtx9xuvjy9hzh4ccfz.streamlit.app/)**

*(Note: free-tier Streamlit apps sleep after inactivity — if the link seems slow to load, give it a few seconds to wake up.)*

![App Screenshot](demo_screenshot.png)

## 📌 Project Overview

This is a capstone project that goes beyond a single model — it's a full **customer intelligence pipeline** covering four connected ML tasks on real telecom customer data:

1. **Churn Classification** — predicting which customers are likely to leave
2. **Tenure Regression** — estimating how long a customer will stay
3. **Customer Segmentation (Clustering)** — grouping customers into behavioral segments
4. **Anomaly Detection** — flagging unusual customer records that don't fit expected patterns

All four models are trained, evaluated, saved, and served together through one Streamlit interface.

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Logistic Regression, Random Forest, K-Means), XGBoost, Isolation Forest (anomaly detection)
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Database:** MySQL (via SQLAlchemy) — pipeline stores cleaned and processed data in a relational database
- **Deployment:** Streamlit Community Cloud
- **Model persistence:** joblib

## 🧹 Data Pipeline

The project processes the Telco Customer Churn dataset through several stages, each saved as a checkpoint:

1. **Cleaning** — handling missing values, correcting data types
2. **Feature engineering** — deriving new features (e.g., tenure groups), encoding categorical variables
3. **MySQL integration** — loading processed data into a MySQL database table for persistent, queryable storage
4. **Clustering** — K-Means segmentation with scaled features, cluster labels added back to the dataset
5. **Anomaly detection** — Isolation Forest flags outlier customer records
6. Final dataset (with cluster labels and anomaly flags) saved and pushed back into MySQL

## 🤖 Models

| Task | Models Compared | Purpose |
|---|---|---|
| Churn Classification | Logistic Regression, Random Forest, XGBoost | Predict Yes/No churn likelihood |
| Tenure Regression | Linear Regression (best performer) | Estimate expected customer tenure |
| Segmentation | K-Means (with optimal K selection) | Group customers into behavioral clusters |
| Anomaly Detection | Isolation Forest | Flag statistically unusual customer records |

Each model was evaluated using appropriate metrics (accuracy, precision, recall, F1, ROC-AUC for classification; standard regression metrics for tenure) and the best-performing version was saved for deployment.

## 📊 Key Visualizations

The project includes exploratory and evaluative charts:
- Churn distribution and churn by contract type / tenure
- Monthly charges vs. churn relationship
- Confusion matrix and ROC curves for the classifier
- Feature importance rankings
- Optimal cluster count (elbow method) and PCA-projected cluster visualization
- Anomaly detection results

## 📂 Project Structure

```
customer-intelligence-system/
├── app.py                        # Streamlit app — loads all models, serves predictions
├── work.ipynb                    # Full analysis notebook (cleaning → modeling → deployment)
├── churn_classifier.pkl          # Trained churn classification model
├── feature_columns.pkl           # Feature list for churn model
├── tenure_regressor.pkl          # Trained tenure regression model
├── regression_features.pkl       # Feature list for regression model
├── kmeans_cluster.pkl            # Trained K-Means clustering model
├── cluster_scaler.pkl            # Scaler used for clustering features
├── anomaly_detector.pkl          # Trained Isolation Forest model
├── telco_churn_final_complete.csv # Final processed dataset (cleaned + clustered + anomaly-flagged)
├── requirements.txt
└── README.md
```

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dataset

Telco Customer Churn dataset — 7,043 customer records with 21 features covering demographics, account information, and service usage.

## 👤 Author

**Tejas** — Aspiring Data Analyst / Data Scientist
