### Telecom Customer Churn Analytics & Prediction (PPP Framework)

## Live link: 


https://churn-prediction-telecom.onrender.com

## View MLflow Experiments


https://dagshub.com/ranesh88/churn_prediction_telecom.mlflow/#/experiments/0


https://dagshub.com/ranesh88/churn_prediction_telecom.mlflow/#/experiments/0/runs/1c20084de7614a0a80922dc3bb9fb1dc


https://dagshub.com/ranesh88/churn_prediction_telecom.mlflow/#/experiments/0/runs/b8b949076ed5489fbebcc60a7f54cef5


## 📌 Project Overview

Customer churn is a critical challenge in the telecom industry, directly impacting revenue and customer lifetime value.
This project aims to analyze, understand, and predict customer churn using an end-to-end analytics workflow based on the PPP framework:

PPP = PostgreSQL + Python + Power BI

Rather than focusing only on prediction, the core objective is to identify churn drivers, quantify their impact, and communicate actionable insights for business decision-making.

## 🎯 Business Objectives

Identify key factors influencing customer churn

Segment customers based on churn risk

Support data-driven retention strategies

Provide stakeholders with interactive dashboards

Demonstrate an end-to-end analytics pipeline used in real organizations

## 🧱 Tech Stack (PPP)
Layer	Tools
Database	PostgreSQL
Data Analysis	Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels)
Modeling	Scikit-learn
Visualization	Power BI
Deployment (Optional)	FastAPI, Streamlit
Version Control	Git & GitHub
## 🗂️ Data Architecture

Although the raw dataset was flat, a relational schema was designed to reflect real-world telecom systems.

Example logical tables:

customers

contracts

billing

service_usage

churn_labels

This enabled:

Primary & foreign key relationships

SQL joins and aggregations

Better data validation and scalability

### 🔄 Project Workflow
## 1️⃣ Data Extraction (PostgreSQL)

Designed schema and loaded data into PostgreSQL

Wrote SQL queries to:

Join multiple tables

Validate data granularity

Generate analytical datasets

Documented queries with business insights

## 2️⃣ Exploratory Data Analysis (Python)
Data Quality & Preparation

Feature type classification

Data type optimization

Missing value treatment

Outlier detection (IQR, violin plots)

Unique value & consistency checks

Univariate & Bivariate Analysis

Distribution analysis (skewness & kurtosis)

Frequency analysis for categorical variables

Churn rate vs customer volume by segment

Pairwise relationships

Statistical Analysis

Correlation matrix & multicollinearity checks

Chi-square tests for categorical features

p-value and degree of freedom interpretation

Target variable balance analysis

## 3️⃣ Feature Engineering & Modeling

Feature transformation and encoding

Data preprocessing for modeling

Multiple classification models evaluated:

Logistic Regression

Tree-based models

Model comparison using appropriate metrics

Selection of best model based on performance and interpretability

Feature importance analysis to explain churn drivers

⚠️ Note:
The model is used as a decision-support tool, not the primary outcome.

## 4️⃣ Business Insights & Visualization (Power BI)

Designed interactive Power BI dashboards:

Overall churn trends

Churn by contract, tenure, and services

High-risk customer segments

Focused on storytelling, not just charts

Insights aligned with retention strategy recommendations

## 5️⃣ Optional Productionization (Value-Add)

(Included to demonstrate end-to-end understanding)

Model serialization using Joblib

Lightweight API using FastAPI / Streamlit

Dockerized application

Experiment tracking using MLflow / DagsHub

CI/CD with GitHub Actions

Public deployment for demonstration purposes

## 📈 Key Insights (Sample)

Month-to-month contracts show significantly higher churn

Short tenure customers are at the highest risk

Billing method and service usage patterns strongly influence churn

Statistical tests confirm several categorical variables have a significant relationship with churn

## 📊 Deliverables

Clean, documented SQL queries

Python notebooks for EDA & modeling

Power BI interactive dashboard

README & project documentation

Public demo link (if deployed)

## 🧠 What This Project Demonstrates

Strong SQL & relational data modeling

Applied statistical thinking

Practical Python data analysis

Business-focused Power BI storytelling

Ability to work end-to-end like a real analyst

## 🚀 Future Enhancements

Incorporate time-series churn trends

Add customer lifetime value (CLV) estimation

Automate data refresh in Power BI

Extend analysis to pricing optimization

## 📌 Author Note

This project follows a real-world analytics workflow rather than a tutorial-driven approach.
The emphasis is on insights, decision support, and communication, not just model accuracy.



