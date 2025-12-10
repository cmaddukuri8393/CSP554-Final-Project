# Electricity Consumption Forecasting using Big Data Technologies  
CSP 554 Final Project  
**Illinois Institute of Technology**  
**Team Members:** Meghana Rabba, Rohan Singh  
**Instructor:** Prof. Joseph Rosen  

---

## 1. Abstract  
This project presents a complete Big Data forecasting pipeline designed to process multi year, high resolution electricity consumption data across five U.S. regions. Using Amazon S3 for distributed storage and Amazon EMR with Apache Spark for computation, the pipeline performs end to end ingestion, preprocessing, feature engineering, model training, and result generation. Advanced machine learning models including Random Forest, Gradient Boosting, XGBoost, and LightGBM were evaluated. The results demonstrate strong forecasting performance, confirming the effectiveness of a cloud based, scalable architecture for large scale time series modeling.  
:contentReference[oaicite:1]{index=1}

---

## 2. Introduction  
Electricity load forecasting is essential for power system planning and grid stability. Traditional environments fail to handle the scale and complexity of modern electricity datasets. This project uses a cloud native Big Data architecture integrating:  
• **Amazon S3** for distributed storage  
• **Amazon EMR** for scalable cluster based computation  
• **Apache Spark** for fast parallel preprocessing  
• **Machine learning models** for forecasting  

This pipeline processes millions of hourly load records across AEP, COMED, DAYTON, DOM, and PJM regions.  
:contentReference[oaicite:2]{index=2}

---

## 3. Objectives  
The project aims to build a scalable forecasting system capable of:  
1. Organizing multi region datasets in S3 using partitioned Parquet format.  
2. Preprocessing data using PySpark including cleaning, transformation, and time series feature engineering.  
3. Training advanced ML models such as Random Forest, Gradient Boosting, XGBoost, LightGBM.  
4. Evaluating model performance using RMSE, MAE, R².  
5. Demonstrating distributed Big Data processing using EMR and Spark.  
:contentReference[oaicite:3]{index=3}

---

## 4. Scope  
The pipeline focuses on batch forecasting using cloud environments. Key tasks include:  
• End to end pipeline creation using S3 and EMR  
• Multi region ingestion, cleaning, and feature engineering  
• ML training and evaluation  
• Automated region wise outputs  
Excluded: deep learning, real time systems, production deployment.  
:contentReference[oaicite:4]{index=4}

---

## 5. Dataset Overview  
The dataset consists of hourly consumption records stored as Parquet files in S3, partitioned as:  
region = AEP / COMED / DAYTON / DOM / PJM
year = 1998 … 2023
month = 1 … 12


Each file contains:  
• timestamp  
• load_mw  
• temperature, humidity, wind speed  
• lagged load features (1h, 24h, 7d)  
• rolling averages  
• holiday and weekend indicators  

Each region exceeds 800,000 rows after feature engineering.  
:contentReference[oaicite:5]{index=5}

---

## 6. Significance  
The project showcases how Big Data ecosystems enable:  
• Accurate large scale forecasting  
• Distributed feature engineering  
• Cloud based scalable compute  
• Transforming raw massive datasets into usable actionable insights  
:contentReference[oaicite:6]{index=6}

---

## 7. Literature Review  
Recent studies show ensemble ML methods such as Random Forest and Gradient Boosting outperform classical methods like ARIMA for electricity forecasting, especially on nonlinear, seasonal datasets. Additionally, Big Data frameworks such as Spark and cloud storage systems (S3) are now essential for high resolution time series analysis.  
:contentReference[oaicite:7]{index=7}

---

## 8. Data Preprocessing  
Using PySpark on EMR, preprocessing steps included:  
1. Loading parquet files from S3  
2. Converting all fields to numeric types  
3. Extracting time based features  
4. Handling missing values (interpolation and forward/back fill)  
5. Creating lagged features  
6. Constructing rolling averages  
7. Adding interaction features  
8. Encoding categorical fields  
9. Reshaping sequences for ML input  
:contentReference[oaicite:8]{index=8}

---

## 9. Tools Used  
• Amazon S3  
• Amazon EMR (Hadoop + Spark cluster)  
• Apache Spark (PySpark)  
• Python  
• Scikit learn  
• XGBoost and LightGBM  
• H2O AutoML (locally tested)  
• Matplotlib, Seaborn  
• AWS CLI, SSH  
:contentReference[oaicite:9]{index=9}

---

## 10. Exploratory Data Analysis (EDA)  
EDA was conducted per region, including:  
• Hourly load profiling  
• Temperature vs load scatter analysis  
• Correlation heatmaps  
• Lag dependency inspection  
• Rolling means visualization  

Key insights from AEP (example):  
• Load peaks between 5 PM and 8 PM  
• Strong daily and weekly seasonality  
• U shaped temperature demand relationship  
:contentReference[oaicite:10]{index=10}

---

## 11. Preprocessing and Modelling Workflow  

### 11.1 Feature Engineering  
Weather, timestamp, cyclical encodings, lagged loads, rolling statistics, and interaction features were created.  
:contentReference[oaicite:11]{index=11}

### 11.2 Handling Missing Values  
Interpolation, forward fill, and backward fill ensured time series continuity.  
:contentReference[oaicite:12]{index=12}

### 11.3 Scaling  
StandardScaler applied to all numerical features prior to sequence creation.  
:contentReference[oaicite:13]{index=13}

### 11.4 Sequence Creation  
A 12 hour input window predicts the next hour:  
:contentReference[oaicite:14]{index=14}

### 11.5 Models Implemented  
1. Linear Regression  
2. Ridge  
3. Lasso  
4. Random Forest  
5. Gradient Boosting  
6. XGBoost  
7. LightGBM  
8. H2O AutoML  
:contentReference[oaicite:15]{index=15}

### 11.6 Train Test Split  
• 80 percent training  
• 20 percent testing  
Evaluation metrics: RMSE, MAE, R²  
:contentReference[oaicite:16]{index=16}

---

## 12. Analysis of Results  

### 12.1 Region AEP  
**Best Model: XGBoost**  
• RMSE = 262.89  
• R² = 0.9889  
• MAE = 160.03  
:contentReference[oaicite:17]{index=17}

### 12.2 Region COMED  
**Best Model: XGBoost**  
• RMSE = 184.861  
• R² = 0.993  
• MAE = 117.615  
:contentReference[oaicite:18]{index=18}

### 12.3 Region DOM  
**Best Model: XGBoost**  
• RMSE = 212.532  
• R² = 0.993  
• MAE = 130.822  
:contentReference[oaicite:19]{index=19}

### 12.4 Region PJM  
**Best Model: Random Forest**  
• RMSE = 0.000  
• R² = 1.00  
• MAE = 0.000  
:contentReference[oaicite:20]{index=20}

### 12.5 Region DAYTON  
**Best Model: XGBoost**  
DAYTON showed extremely high predictability and consistent cyclic behavior.  
:contentReference[oaicite:21]{index=21}

---

## 13. Big Data Architecture  

### 13.1 Amazon S3 Storage  
Datasets stored in S3 using region → year → month partitioning.  
Over **900,000+ hourly records** stored.  
:contentReference[oaicite:22]{index=22}

### 13.2 Amazon EMR for Distributed Computing  
EMR executed:  
• Feature engineering  
• Sequence generation  
• Model training  
• Distributed I/O using Spark  
:contentReference[oaicite:23]{index=23}

### 13.3 Automated Multi Region Pipeline  
Pipeline automatically:  
1. Detects available regions  
2. Loads data  
3. Performs EDA  
4. Engineers features  
5. Creates sequences  
6. Trains seven ML models  
7. Stores outputs  
:contentReference[oaicite:24]{index=24}

### 13.4 Spark on EMR  
Example PySpark loading:  
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadFeatures").getOrCreate()
df = spark.read.parquet("s3://bigdata-electricity-features-meghanarabba/region=AEP/")
df.printSchema()
```
### 13.4 Spark on EMR 
This project builds a complete Big Data forecasting system for multi region electricity load using Amazon S3, Amazon EMR, and Apache Spark. All regional datasets are stored in partitioned Parquet format on S3 and processed in parallel on EMR using PySpark for cleaning, feature engineering, lag creation, rolling statistics, and sequence generation. Multiple machine learning models including Random Forest, Gradient Boosting, XGBoost, and LightGBM were trained, with XGBoost consistently delivering the strongest performance across most regions. The system automatically performs EDA, preprocessing, modeling, evaluation, and output generation for each region, demonstrating a scalable end to end pipeline capable of handling millions of time series records. This architecture proves the efficiency of cloud based distributed processing for large scale forecasting tasks, producing accurate, interpretable, and region specific insights suitable for real world energy demand planning.
