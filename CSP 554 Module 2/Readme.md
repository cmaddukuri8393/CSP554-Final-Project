# Module 2 – Deep Learning (Single Script)

Script: `LSTM and GRU.py`
This script does:

* Load features from S3
* Run EDA + save CSVs to S3
* Generate plots on EMR
* Train LSTM & GRU
* Print MAE/RMSE and save metrics

---

## 1. SSH into EMR Master

```bash
ssh -i <your-keypair>.pem hadoop@<ec2-public-dns>
```

Example:

```bash
ssh -i emr-key-pair.pem hadoop@ec2-18-118-110-20.us-east-2.compute.amazonaws.com
```

---

## 2. Go to Project Directory

```bash
cd ~/electricity_forecast
```

Make sure `LSTM and GRU.py` is in this folder.

---

## 3. (Optional) Create & Activate venv

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Python Packages

On EMR master (inside venv if using):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow==2.12.0 opt-einsum
```

---

## 5. Verify TensorFlow (optional)

```bash
python3 -c "import tensorflow as tf; print('TF OK', tf.__version__)"
```

Result:

```text
TF OK 2.12.0
```

---

## 6. Run the Pipeline Script

Use Spark to run the script:

```bash
spark-submit LSTM and GRU.py
```

This will:

* Read features from:

  ```text
  S3_FEATURES_PATH = "s3://csp554-electricityforecast/features/features/"
  ```

* Write EDA CSVs to:

  ```text
  s3://csp554-electricityforecast/outputs/eda_*
  ```

* Write histogram & missing values to:

  ```text
  s3://csp554-electricityforecast/outputs/eda_histogram
  s3://csp554-electricityforecast/outputs/eda_missing_values
  ```

* Save plots locally to:

  ```text
  ~/electricity_forecast/plots/
  ```

* Train LSTM & GRU and print something like:

  ```text
  LSTM_fast - MAE: 550.xx MW, RMSE: 691.xx MW
  GRU_fast  - MAE: 559.xx MW, RMSE: 701.xx MW
  ```

* Save metrics CSV:

  ```text
  ~/electricity_forecast/plots/model_metrics.csv
  ```

---

## 7. Check Outputs on EMR

### 7.1 Local plots & metrics

```bash
ls ~/electricity_forecast/plots
```

You should see:

* `01_load_distribution.png`
* `02_hourly_pattern.png`
* `03_dow_pattern.png`
* `04_load_vs_temperature.png`
* `05_correlation_heatmap.png`
* `06_true_vs_pred_test.png`
* `model_metrics.csv`

### 7.2 EDA CSVs in S3

```bash
aws s3 ls s3://csp554-electricityforecast/outputs/
```

Optional (download locally to EMR):

```bash
aws s3 sync s3://csp554-electricityforecast/outputs ~/electricity_forecast/outputs_from_s3
```

---

## 8. Download Results to local machine

From your **local machine** (not EMR):

move to your pwd in Terminal:

```bash
chmod 400 emr-key-pair.pem
scp -i emr-key-pair.pem -r hadoop@ec2-18-118-110-20.us-east-2.compute.amazonaws.com:~/electricity_forecast/plots ./plots_module2
```

Now you have all the plots and `model_metrics.csv` locally for your report.

---

## 9. If You Change Bucket or Paths

Edit these lines at the top of `LSTM and GRU.py`:

```python
S3_FEATURES_PATH = "s3://csp554-electricityforecast/features/features/"
OUTPUT_BASE      = "s3://csp554-electricityforecast/outputs"
```

Then re-run:

```bash
spark-submit LSTM and GRU.py
```
---
