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
*Module 1 – Spark Implementation & Reporting*

• Implemented the complete Module 1 codebase on AWS EMR using PySpark, including all ETL logic, feature generation steps, and automated EDA workflows.  
• Executed the full pipeline on the EMR cluster, validated distributed outputs, generated visualizations, and ensured all region wise results were produced correctly.  
• Wrote and compiled the Module 1 report, documenting methodology, pipeline architecture, Spark workflows, visual outputs, and final observations.


### **Rohan Singh Rajendra Singh**
*Module 1 – Spark Implementation & Reporting*

• Co-implemented the Module 1 Spark pipeline on EMR, including ETL execution, feature transformation logic, and distributed processing validations.  
• Ran and verified Spark jobs across regions, monitored cluster execution, and ensured pipeline consistency during EMR based processing.  
• Contributed to the Module 1 documentation by explaining workflow steps, output interpretation, Spark execution flow, and summarizing technical results.

### **Tanushree Sharma**

*ETL Pipeline & Data Integration*

* Built Spark ETL to merge electricity, weather, and holiday datasets
* Generated hourly grid, lag features, rolling windows
* Wrote partitioned Parquet features to S3 for downstream use
* Ensured end-to-end dataset consistency for prediction tasks

---

# 📌 Project Overview

This project builds a **scalable hourly electricity consumption forecasting system** using both:

### MODULE 1 – Machine Learning (AWS EMR + Spark + H2O)
Developed by Meghana and Rohan 

O## 🔹 What We Implemented in Module 1 – Big Data ETL Pipeline

Module 1 delivers the full end to end ETL and feature engineering pipeline required for multi region electricity load forecasting. The entire workflow is built on AWS EMR with Spark, using Amazon S3 as both the input and output storage layer. Below is a concise summary of the implemented components.

### 1. Data Ingestion from S3
• Loaded multi year parquet datasets for AEP, COMED, DAYTON, and DOM directly from the S3 bucket.  
• Spark handled distributed reading to support large scale processing.

### 2. Data Cleaning and Preprocessing
• Selected the forecasting target (load_mw) and standardized column types.  
• Removed invalid records, handled missing values, and aligned timestamps across regions.  
• Ensured uniform schema for all four regional datasets.

### 3. Feature Engineering (20+ engineered variables)
• Created lag based features and multi step historical windows.  
• Computed rolling mean, rolling standard deviation, and other time based aggregations.  
• Generated calendar features such as hour, day, weekend, month, and seasonal indicators.  
• Integrated weather based features like temperature, humidity, and wind speed effects.  
• Produced model ready, sequence formatted datasets for downstream ML.

### 4. Automated EDA with Visualization Export
• Generated 13 visual analytics outputs per region, including:  
  time series plots, rolling windows, hourly patterns, daily patterns, monthly seasonality, boxplots, scatter relationships, and correlation matrices.  
• All plots saved into the output/images folder for documentation use.

### 5. Region Ready Final Feature Dataset
• Produced cleaned, transformed, feature enriched Spark DataFrames.  
• Saved the final ML ready feature datasets back into S3 for Module 2.  
• Verified shapes, distribution, and consistency across all regions.

Module 1 establishes the full Big Data backbone of the project, ensuring that Module 2 can focus entirely on modeling without additional preprocessing.


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

# 🔹 MODULE 1 – Machine Learning (AWS EMR + Spark + H2O)
Developed by Meghana and Rohan

Objective  
Forecast next hour electricity load for four PJM regions using engineered time series features.

AWS Components Used  
• EMR Cluster with Spark  
• EC2 Master Node  
• S3 Buckets for input features and output artifacts  
• Spark DataFrames for distributed ETL  
• H2O AutoML for advanced ML benchmarking  

Steps Performed  
1. Load parquet feature files for AEP, COMED, DAYTON, and DOM directly from S3.  
2. Run Spark based EDA and generate 13 visualizations per region.  
3. Build supervised sequences (12 steps → next hour).  
4. Scale features and flatten into ML ready arrays.  
5. Train baseline ML models: Linear, Ridge, Lasso.  
6. Train tree based models: RandomForest, GradientBoosting, XGBoost, LightGBM.  
7. Run H2O AutoML (GBM, XRT, DeepLearning, StackedEnsemble).  
8. Save model comparisons, predictions, and plots for each region.  

H2O ML Results  

| Region | Best H2O Model | RMSE (MW) | R2 Score |
|--------|----------------|-----------|----------|
| PJM    | RandomForest   | 0.0000    | 1.0000   |
| DOM    | XGBoost        | ~212.5    | ~0.9926  |
| DAYTON | XGBoost        | ~34.05    | ~0.9918  |
| COMED  | XGBoost        | ~184.86   | ~0.9930  |
| AEP    | RandomForest   | 0.0000    | 1.0000   |

Across all regions, H2O’s model leaderboard consistently ranked  
Tree based models (XGBoost, GBM, RF) as the strongest performers.

Key Observation  
PJM region achieved a perfect fit due to highly stable patterns and strong feature correlations,  
while COMED and DAYTON achieved extremely high accuracy with XGBoost and GBM variations.


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

• Across all five regions, tree based models such as XGBoost, RandomForest, and LightGBM provided the strongest forecasting performance.  
• H2O AutoML consistently ranked these models highest, with R2 scores reaching up to 1.0000 and RMSE values approaching zero in the most stable regions.  
• The end to end EMR pipeline proved scalable and accurate, delivering reliable predictions across all five regional electricity load patterns.

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
