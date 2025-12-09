#!/usr/bin/env python3
import os

# Use non-interactive backend for matplotlib (no GUI on EMR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # ============================================================
    # 0. Config: S3 paths & local output dir
    # ============================================================
    TIME_COL   = "timestamp"
    TARGET_COL = "load_mw"
    REGION_COL = "region"

    # Your actual bucket + prefixes
    S3_FEATURES_PATH = "s3://csp554-electricityforecast/features/features/"
    OUTPUT_BASE      = "s3://csp554-electricityforecast/outputs"

    LOCAL_PLOTS_DIR = os.path.expanduser("~/electricity_forecast/plots")
    os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)

    # ============================================================
    # 1. Start Spark session
    # ============================================================
    spark = (
        SparkSession.builder
        .appName("ElectricityForecast_EDA_LSTM_GRU_Option3")
        .getOrCreate()
    )

    print("Spark version:", spark.version)

    # ============================================================
    # 2. Load parquet from S3
    # ============================================================
    print("Reading parquet from:", S3_FEATURES_PATH)
    df = spark.read.parquet(S3_FEATURES_PATH)

    print("Schema:")
    df.printSchema()
    total_rows = df.count()
    print("Total rows:", total_rows)
    df.show(5, truncate=False)

    # Ensure timestamp type
    df = df.withColumn(TIME_COL, F.to_timestamp(F.col(TIME_COL)))

    # ============================================================
    # 3. Time features & interactions
    # ============================================================
    df_time = (
        df.withColumn("hour",  F.hour(F.col(TIME_COL)))
          .withColumn("dow",   F.dayofweek(F.col(TIME_COL)))  # 1=Sun,...,7=Sat
          .withColumn("day",   F.date_trunc("day", F.col(TIME_COL)))
          .withColumn("week",  F.weekofyear(F.col(TIME_COL)))
          .withColumn("month", F.month(F.col(TIME_COL)))
          .withColumn("is_weekend", F.when(F.col("dow").isin(1, 7), 1).otherwise(0))
          .withColumn("temp_x_weekend", F.col("temperature") * F.col("is_weekend"))
          .withColumn("temp_x_holiday", F.col("temperature") * F.col("is_holiday"))
    )

    df_time.select(
        TIME_COL, "hour", "dow", "day", "week", "month",
        REGION_COL, TARGET_COL, "is_weekend"
    ).show(5, truncate=False)

    # ============================================================
    # 4. EDA: summary, patterns, missing values, correlations
    # ============================================================

    # 4.1 Summary per region
    summary_region = (
        df_time.groupBy(REGION_COL)
               .agg(
                   F.mean(TARGET_COL).alias("avg_load"),
                   F.min(TARGET_COL).alias("min_load"),
                   F.max(TARGET_COL).alias("max_load"),
                   F.stddev(TARGET_COL).alias("std_load"),
                   F.count("*").alias("num_rows")
               )
    )
    print("Summary per region:")
    summary_region.show(truncate=False)

    # 4.2 Hourly, daily, weekly, day-of-week patterns
    hourly_pattern = (
        df_time.groupBy("hour")
               .agg(F.mean(TARGET_COL).alias("avg_load"))
               .orderBy("hour")
    )
    daily_pattern = (
        df_time.groupBy("day")
               .agg(F.mean(TARGET_COL).alias("avg_load"))
               .orderBy("day")
    )
    weekly_pattern = (
        df_time.groupBy("week")
               .agg(F.mean(TARGET_COL).alias("avg_load"))
               .orderBy("week")
    )
    dow_pattern = (
        df_time.groupBy("dow")
               .agg(F.mean(TARGET_COL).alias("avg_load"))
               .orderBy("dow")
    )

    print("Hourly pattern (head):")
    hourly_pattern.show(24, truncate=False)
    print("DOW pattern (head):")
    dow_pattern.show(7, truncate=False)

    # 4.3 Save EDA tables to S3
    for name, d in [
        ("summary_per_region", summary_region),
        ("hourly_pattern",     hourly_pattern),
        ("daily_pattern",      daily_pattern),
        ("weekly_pattern",     weekly_pattern),
        ("dow_pattern",        dow_pattern),
    ]:
        out_path = f"{OUTPUT_BASE}/eda_{name}"
        print(f"Writing {name} to {out_path}")
        (
            d.coalesce(1)
             .write.mode("overwrite")
             .option("header", "true")
             .csv(out_path)
        )

    # 4.4 Histogram of load in Spark
    min_max = df_time.select(
        F.min(TARGET_COL).alias("min_load"),
        F.max(TARGET_COL).alias("max_load")
    ).collect()[0]

    min_load = float(min_max["min_load"])
    max_load = float(min_max["max_load"])
    num_bins = 20
    bin_width = (max_load - min_load) / num_bins

    df_hist = df_time.withColumn(
        "bin_index",
        (((F.col(TARGET_COL) - min_load) / bin_width)).cast("int")
    )

    df_hist = df_hist.withColumn(
        "bin_index",
        F.when(F.col("bin_index") < 0, 0)
         .when(F.col("bin_index") >= num_bins, num_bins - 1)
         .otherwise(F.col("bin_index"))
    )

    hist_counts = (
        df_hist.groupBy("bin_index")
               .agg(F.count("*").alias("count"))
               .orderBy("bin_index")
    )

    print("Histogram counts:")
    hist_counts.show(num_bins, truncate=False)

    hist_out = f"{OUTPUT_BASE}/eda_histogram"
    print(f"Writing histogram to {hist_out}")
    (
        hist_counts.coalesce(1)
                   .write.mode("overwrite")
                   .option("header", "true")
                   .csv(hist_out)
    )

    # 4.5 Missing values
    missing_stats = []
    for c in df_time.columns:
        missing = df_time.select(F.sum(F.col(c).isNull().cast("int")).alias("missing")).collect()[0]["missing"]
        missing_stats.append((c, missing))

    missing_df = spark.createDataFrame(missing_stats, ["column", "missing_count"])
    print("Missing values per column:")
    missing_df.show(truncate=False)

    missing_out = f"{OUTPUT_BASE}/eda_missing_values"
    print(f"Writing missing values to {missing_out}")
    (
        missing_df.coalesce(1)
                  .write.mode("overwrite")
                  .option("header", "true")
                  .csv(missing_out)
    )

    # 4.6 Correlations overall
    print("Global correlations with weather:")
    for col_name in ["temperature", "humidity", "wind_speed"]:
        corr_val = df_time.select(F.corr(TARGET_COL, col_name).alias("corr")).collect()[0]["corr"]
        print(f"corr({TARGET_COL}, {col_name}) = {corr_val}")

    # 4.7 Correlations per region
    regions = [r[REGION_COL] for r in df_time.select(REGION_COL).distinct().collect()]

    for region in regions:
        print(f"\n=== Region: {region} ===")
        df_r = df_time.filter(F.col(REGION_COL) == region)
        for col_name in ["temperature", "humidity", "wind_speed"]:
            corr_val = df_r.select(F.corr(TARGET_COL, col_name)).collect()[0][0]
            print(f"  corr({TARGET_COL}, {col_name}) = {corr_val}")

    # ============================================================
    # 5. EDA Plots (Spark -> Pandas -> PNG)
    # ============================================================
    print("Creating EDA plots...")

    pdf_sample = df_time.sample(0.02).toPandas()
    pdf_sample[TIME_COL] = pd.to_datetime(pdf_sample[TIME_COL])

    # 5.1 Load distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(pdf_sample[TARGET_COL], bins=40, kde=True)
    plt.title("Load Distribution (MW)")
    plt.xlabel("Load (MW)")
    plt.ylabel("Frequency")
    plt.grid(True)
    out_path = os.path.join(LOCAL_PLOTS_DIR, "01_load_distribution.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    # 5.2 Hourly pattern
    hourly_pd = hourly_pattern.toPandas()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="hour", y="avg_load", data=hourly_pd)
    plt.title("Average Load by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average Load (MW)")
    plt.grid(True)
    out_path = os.path.join(LOCAL_PLOTS_DIR, "02_hourly_pattern.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    # 5.3 Day-of-week pattern
    dow_pd = dow_pattern.toPandas()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="dow", y="avg_load", data=dow_pd)
    plt.title("Average Load by Day of Week")
    plt.xlabel("Day of Week (1=Sunday)")
    plt.ylabel("Average Load (MW)")
    plt.grid(True)
    out_path = os.path.join(LOCAL_PLOTS_DIR, "03_dow_pattern.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    # 5.4 Load vs temperature
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=pdf_sample["temperature"], y=pdf_sample[TARGET_COL], alpha=0.3)
    plt.title("Load vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Load (MW)")
    out_path = os.path.join(LOCAL_PLOTS_DIR, "04_load_vs_temperature.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    # 5.5 Correlation heatmap
    heat_cols = [
        "temperature", "humidity", "wind_speed",
        "lag_1h", "lag_24h", "lag_7d", "rolling_mean_24h",
        TARGET_COL
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(pdf_sample[heat_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    out_path = os.path.join(LOCAL_PLOTS_DIR, "05_correlation_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    # ============================================================
    # 6. Deep Learning Prep (single region subset)
    # ============================================================
    DL_FEATURES = [
        "hour", "dow",
        "lag_1h", "lag_24h", "lag_7d",
        "rolling_mean_24h",
        "temperature", "humidity", "wind_speed",
        "is_holiday", "is_weekend",
        "temp_x_weekend", "temp_x_holiday",
    ]

    REGION_TO_USE = "AEP"  # change if needed

    df_dl = (
        df_time
        .filter(F.col(REGION_COL) == REGION_TO_USE)
        .select(TIME_COL, REGION_COL, TARGET_COL, *DL_FEATURES)
        .orderBy(TIME_COL)
    )

    print("Rows in df_dl:", df_dl.count())
    df_dl.show(10, truncate=False)

    pdf_dl = df_dl.toPandas()
    pdf_dl[TIME_COL] = pd.to_datetime(pdf_dl[TIME_COL])
    pdf_dl = pdf_dl.sort_values(TIME_COL).reset_index(drop=True)

    print("Rows in pdf_dl (before slice):", len(pdf_dl))

    # Optional: limit to last N rows to speed up training
    N_RECENT = 100_000
    if len(pdf_dl) > N_RECENT:
        pdf_dl = pdf_dl.tail(N_RECENT).reset_index(drop=True)
        print("Using last", N_RECENT, "rows for DL")

    print("Rows in pdf_dl (after slice):", len(pdf_dl))

    # ============================================================
    # 7. Clean in Pandas (numeric + missing)
    # ============================================================
    feature_cols = DL_FEATURES
    target_col   = TARGET_COL

    pdf_model = pdf_dl[[TIME_COL, target_col] + feature_cols].copy()

    # Ensure numeric
    for c in feature_cols + [target_col]:
        pdf_model[c] = pd.to_numeric(pdf_model[c], errors="coerce")

    # Fill feature NaNs with mean
    for c in feature_cols:
        pdf_model[c] = pdf_model[c].fillna(pdf_model[c].mean())

    # Fill target
    pdf_model[target_col] = (
        pdf_model[target_col]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    pdf_model = pdf_model.dropna(subset=[target_col]).reset_index(drop=True)

    print("Rows after DL cleaning:", len(pdf_model))
    print(pdf_model.dtypes)

    # ============================================================
    # 8. Build sequences (24h -> next hour)
    # ============================================================
    SEQ_LEN = 24
    HORIZON = 1

    values = pdf_model[feature_cols].astype("float32").values
    target_arr = pdf_model[target_col].astype("float32").values

    X_list, y_list = [], []
    max_start = len(pdf_model) - SEQ_LEN - HORIZON + 1

    for start in range(max_start):
        end = start + SEQ_LEN
        y_idx = end + HORIZON - 1
        X_list.append(values[start:end])
        y_list.append(target_arr[y_idx])

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ============================================================
    # 9. Train/Val/Test split + scaling
    # ============================================================
    n = len(X)
    train_size = int(0.7 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test,  y_test  = X[train_size+val_size:], y[train_size+val_size:]

    num_features = X.shape[-1]

    scaler_x = StandardScaler()

    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat   = X_val.reshape(-1, num_features)
    X_test_flat  = X_test.reshape(-1, num_features)

    X_train_scaled = scaler_x.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled   = scaler_x.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled  = scaler_x.transform(X_test_flat).reshape(X_test.shape)

    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_val_scaled   = np.nan_to_num(X_val_scaled)
    X_test_scaled  = np.nan_to_num(X_test_scaled)

    y_train_mean = float(np.mean(y_train))
    y_train_std  = float(np.std(y_train))
    if y_train_std == 0 or np.isnan(y_train_std):
        y_train_std = 1.0

    y_train_scaled = ((y_train - y_train_mean) / y_train_std).astype("float32")
    y_val_scaled   = ((y_val   - y_train_mean) / y_train_std).astype("float32")
    y_test_scaled  = ((y_test  - y_train_mean) / y_train_std).astype("float32")

    print("Train/Val/Test:", X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)
    print("y_train_mean, y_train_std:", y_train_mean, y_train_std)

    # ============================================================
    # 10. Define LSTM & GRU models
    # ============================================================
    def build_lstm_fast(seq_len, num_features):
        inputs = keras.Input(shape=(seq_len, num_features))
        x = layers.LSTM(32, dropout=0.2)(inputs)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs, name="LSTM_fast")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def build_gru_fast(seq_len, num_features):
        inputs = keras.Input(shape=(seq_len, num_features))
        x = layers.GRU(32, dropout=0.2)(inputs)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs, name="GRU_fast")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae"]
        )
        return model

    seq_len = SEQ_LEN
    num_features = X_train_scaled.shape[-1]

    lstm_model = build_lstm_fast(seq_len, num_features)
    gru_model  = build_gru_fast(seq_len, num_features)

    lstm_model.summary()
    gru_model.summary()

    # ============================================================
    # 11. Train models
    # ============================================================
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    EPOCHS = 20
    BATCH_SIZE = 256

    print("Training LSTM...")
    history_lstm = lstm_model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("Training GRU...")
    history_gru = gru_model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ============================================================
    # 12. Evaluate models (denormalized)
    # ============================================================
    def evaluate_model(model, X, y_true_raw, mean, std, name):
        y_pred_scaled = model.predict(X).ravel()
        y_pred = y_pred_scaled * std + mean

        mae = mean_absolute_error(y_true_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred))
        print(f"{name} - MAE: {mae:.2f} MW, RMSE: {rmse:.2f} MW")
        return mae, rmse, y_pred

    print("Evaluating LSTM_fast...")
    mae_lstm, rmse_lstm, y_pred_lstm = evaluate_model(
        lstm_model, X_test_scaled, y_test, y_train_mean, y_train_std, "LSTM_fast"
    )

    print("Evaluating GRU_fast...")
    mae_gru, rmse_gru, y_pred_gru = evaluate_model(
        gru_model, X_test_scaled, y_test, y_train_mean, y_train_std, "GRU_fast"
    )

    results_df = pd.DataFrame([
        {"Model": "LSTM_fast", "MAE_MW": mae_lstm, "RMSE_MW": rmse_lstm},
        {"Model": "GRU_fast",  "MAE_MW": mae_gru,  "RMSE_MW": rmse_gru},
    ])

    print("Results table:")
    print(results_df)

    # Save metrics to CSV locally
    metrics_path_local = os.path.join(LOCAL_PLOTS_DIR, "model_metrics.csv")
    results_df.to_csv(metrics_path_local, index=False)
    print("Saved metrics CSV:", metrics_path_local)

    # ============================================================
    # 13. Plot predictions vs true (best model)
    # ============================================================
    if rmse_lstm <= rmse_gru:
        best_name = "LSTM_fast"
        best_pred = y_pred_lstm
    else:
        best_name = "GRU_fast"
        best_pred = y_pred_gru

    print("Best model:", best_name)

    N_PLOT = min(500, len(y_test))

    plt.figure(figsize=(12, 5))
    plt.plot(range(N_PLOT), y_test[:N_PLOT], label="True Load (MW)", linewidth=1)
    plt.plot(range(N_PLOT), best_pred[:N_PLOT], label=f"Predicted Load (MW) - {best_name}", linewidth=1)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Load (MW)")
    plt.title(f"True vs Predicted Load on Test Set ({best_name})")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(LOCAL_PLOTS_DIR, "06_true_vs_pred_test.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

    print("Pipeline completed successfully.")

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()
