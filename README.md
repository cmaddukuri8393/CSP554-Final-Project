# CSP554-Final-Project

**Electricity Consumption Forecasting**
**CSP 554 – Big Data Technologies**

---

## 👥 Team Members & Responsibilities

### **Chaitanya Datta Maddukuri**

*Module 2 (Deep Learning)*

* Developed LSTM & GRU models
* Implemented full Big Data pipeline using Spark + EMR + EC2 + S3
* Performed EDA, sequence modeling, scaling, and model evaluation
* Managed AWS infrastructure, integration, and automation

### **Meghana Rabba**

*Module 1 (Machine Learning)*

* Developed ML models (Linear Regression, Random Forest, XGBoost/GBT)
* Created training, validation, and evaluation framework
* Performed feature-based ML analysis

### **Rohan Singh Rajendra Singh**

*Module 1 (Data Preparation & Feature Engineering)*

* Cleaned raw data for ML experiments
* Developed preprocessing logic used for ML models
* Contributed to EDA for ML module

### **Tanushree Sharma**

*ETL Pipeline & Data Integration*

* Built Spark ETL to merge electricity, weather, and holiday datasets
* Generated hourly grid, lag features, rolling windows
* Wrote partitioned Parquet features to S3 for downstream use
* Ensured end-to-end dataset consistency for prediction tasks

---

# 📌 Project Overview

This project builds a **scalable hourly electricity consumption forecasting system** using both:

### **Module 1 – Machine Learning (Local environment)**

* Logistic workflows done **locally**, NOT using AWS
* Implemented using Python, Pandas, and Scikit-Learn
* Includes ML models:

  * Linear Regression
  * Random Forest
  * Gradient Boosted Trees (GBT/XGBoost)
* Used cleaned features provided by ETL (Tanushree)

### **Module 2 – Deep Learning (Big Data)**

* Implemented by Chaitanya
* Uses **AWS EMR, EC2, Spark, TensorFlow, S3**
* Includes DL models:

  * LSTM
  * GRU
* Handles millions of records with distributed preprocessing
* Produces EDA, metrics, and forecasting plots

---

# 📦 ETL Pipeline (Developed by Tanushree)

Source code: `electricity-etl.py`

### Responsibilities:

✔ Reads electricity load data for **AEP, DOM, PJM, COMED, DAYTON**
✔ Reads weather data: temperature, humidity, wind speed
✔ Reads national holiday dataset
✔ Converts datetime → timestamp
✔ Builds **complete hourly grid per region**
✔ Creates lag features:

* `lag_1h`
* `lag_24h`
* `lag_7d`

✔ Creates rolling window features:

* `rolling_mean_24h`

✔ Adds calendar features:

* `is_holiday`
* `is_weekend`
* `year`, `month`

✔ Writes processed, partitioned Parquet files to:

```
s3://electricity-forcast-tanushree/processed/features/
```

These Parquet files are the **official dataset** used for Module 2.

---

# 🔹 MODULE 1 – Machine Learning (Local Environment)

**Important:**
➡️ **Module 1 DID NOT use AWS EMR or S3.**
➡️ Work was done locally with cleaned datasets exported from ETL.
➡️ Grading should apply separately for Module 1.

### Tasks:

* Train ML models using Scikit-Learn
* Perform local EDA
* Evaluate performance using MAE, RMSE
* Compare multiple models

---

# 🔹 MODULE 2 – Deep Learning (AWS EMR + Spark + TF)

Developed by **Chaitanya**

### Objective

Predict **next-hour electricity load** using a **24-hour historical window**.

### AWS Components Used

* **EMR Cluster** (Spark, Hadoop)
* **EC2 Master Node**
* **S3 Buckets** (input & output)
* **Spark DataFrames**
* **TensorFlow 2.12**

### Steps Performed

1. Load ETL parquet data from S3
2. Run Spark-based EDA
3. Generate statistical summaries + correlations
4. Convert Spark → Pandas → NumPy
5. Build supervised sequences (24h → next hour)
6. Scale features
7. Train LSTM + GRU
8. Compute MAE & RMSE
9. Save plots + metrics

### DL Model Results

| Model         | MAE (MW) | RMSE (MW) |
| ------------- | -------- | --------- |
| **LSTM_fast** | ~550     | ~690      |
| **GRU_fast**  | ~559     | ~702      |

LSTM performed slightly better.

---

# 📝 FINAL RESULTS SUMMARY

### **Module 1 (ML)**

* Performed locally
* Random Forest & GBT showed strong baseline accuracy
* Useful for benchmarking DL performance

### **Module 2 (DL)**

* Performed on AWS EMR with large-scale distributed preprocessing
* LSTM achieved best performance
* End-to-end automated pipeline created by Chaitanya

---

# 📌 LICENSE

This project is for academic use only (CSP 554 Final Project).

---

# 📧 CONTACT

**Chaitanya Datta Maddukuri (Module 2 Lead)**
Email: [cmaddukuri@hawk.illinoistech.edu](mailto:cmaddukuri@hawk.illinoistech.edu)
