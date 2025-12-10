# ============================================================================
# MULTI REGION PIPELINE WITH H2O AutoML - MATPLOTLIB VERSION
# CSP 554 Electricity Load Forecasting
# ============================================================================

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

import pyarrow.parquet as pq
import joblib
import builtins

# Use matplotlib instead of plotly
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# optional extra models
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# H2O AutoML
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except Exception:
    H2O_AVAILABLE = False
    print("Warning: H2O not available. Install with: pip install h2o")


# ============================================================================
# CONFIG FLAGS
# ============================================================================

USE_GPU = False

# H2O AutoML settings
H2O_MAX_RUNTIME_SECS = 300  # 5 minutes per region
H2O_MAX_MODELS = 20


# ============================================================================
# PATH CONFIG
# ============================================================================

# S3 bucket that holds parquet feature folders
# Expected layout:
#   s3://bigdata-electricity-features-meghanarabba/region=AEP
#   s3://bigdata-electricity-features-meghanarabba/region=COMED
#   s3://bigdata-electricity-features-meghanarabba/region=PJM
#   s3://bigdata-electricity-features-meghanarabba/region=DOM
BASE_PATH = "s3://bigdata-electricity-features-meghanarabba"


class Config:
    BASE = "output"
    MODELS = os.path.join(BASE, "models")
    REPORTS = os.path.join(BASE, "reports")
    IMAGES = os.path.join(BASE, "images")

    @staticmethod
    def init_dirs():
        for d in [Config.BASE, Config.MODELS, Config.REPORTS, Config.IMAGES]:
            os.makedirs(d, exist_ok=True)
        print("Output folders ready")
        print("  Models:", Config.MODELS)
        print("  Reports:", Config.REPORTS)
        print("  Images:", Config.IMAGES)


Config.init_dirs()


# ============================================================================
# SAFE HELPERS
# ============================================================================

def safe_min(*args):
    return builtins.min(args)


def safe_round(x, digits=2):
    try:
        return builtins.round(float(x), digits)
    except Exception:
        return x


# ============================================================================
# MATPLOTLIB EDA ANALYZER
# ============================================================================

class MatplotlibEDAAnalyzer:
    """
    Builds visualizations using matplotlib for better compatibility.
    Saves as PNG images.
    """

    def __init__(self, df, target_col="load_mw", time_col="timestamp", region_name="Region"):
        self.region = region_name
        self.target = target_col
        self.time_col = time_col
        self.df = df.copy()
        self.files = []

        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors="coerce")
            self.df = self.df.sort_values(self.time_col)

    def _save_fig(self, fig, filename):
        """Save figure as PNG"""
        png_path = os.path.join(Config.IMAGES, f"{self.region}_{filename}.png")
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved image:", png_path)
        self.files.append(png_path)
        plt.close(fig)

    def run_all(self):
        print("\n" + "=" * 80)
        print(f"EDA STEP - Running visual analysis for {self.region}")
        print("=" * 80)

        self.time_series_main()
        self.rolling_view()
        self.hourly_pattern()
        self.daily_pattern()
        self.monthly_pattern()
        self.target_distribution()
        self.target_boxplot()
        self.weekday_weekend()
        self.temp_humidity_wind_scatter()
        self.correlation_matrix()
        self.target_correlations_bar()

        print(f"\nEDA complete for {self.region}, images:", len(self.files))
        return self.files

    def time_series_main(self):
        if self.time_col not in self.df.columns or self.target not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df[self.time_col], self.df[self.target], linewidth=0.5)
        ax.set_title(f"{self.region} Time Series of {self.target}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Load MW")
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "01_time_series")

    def rolling_view(self):
        if self.time_col not in self.df.columns or self.target not in self.df.columns:
            return

        df = self.df.copy()
        df["rolling_24"] = df[self.target].rolling(window=24, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df[self.time_col], df[self.target], linewidth=0.5, alpha=0.5, label='Actual')
        ax.plot(df[self.time_col], df["rolling_24"], linewidth=1.5, label='24 Hour Rolling Mean')
        ax.set_title(f"{self.region} Load with 24 Hour Rolling Mean", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Load MW")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "02_rolling_mean")

    def hourly_pattern(self):
        if self.time_col not in self.df.columns or self.target not in self.df.columns:
            return

        df = self.df.copy()
        df["hour"] = df[self.time_col].dt.hour
        hourly = df.groupby("hour")[self.target].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hourly["hour"], hourly[self.target], marker='o', linewidth=2)
        ax.set_title(f"{self.region} Average Load by Hour", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Load MW")
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "03_hourly_pattern")

    def daily_pattern(self):
        if self.time_col not in self.df.columns or self.target not in self.df.columns:
            return

        df = self.df.copy()
        df["dayofweek"] = df[self.time_col].dt.dayofweek
        daily = df.groupby("dayofweek")[self.target].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(daily["dayofweek"], daily[self.target], color='steelblue', alpha=0.7)
        ax.set_title(f"{self.region} Average Load by Day of Week", fontsize=14, fontweight='bold')
        ax.set_xlabel("Day of Week zero Monday")
        ax.set_ylabel("Average Load MW")
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.grid(True, alpha=0.3, axis='y')
        self._save_fig(fig, "04_daily_pattern")

    def monthly_pattern(self):
        if self.time_col not in self.df.columns or self.target not in self.df.columns:
            return

        df = self.df.copy()
        df["month"] = df[self.time_col].dt.month
        monthly = df.groupby("month")[self.target].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly["month"], monthly[self.target], marker='o', linewidth=2, markersize=8)
        ax.set_title(f"{self.region} Average Load by Month", fontsize=14, fontweight='bold')
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Load MW")
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "05_monthly_pattern")

    def target_distribution(self):
        if self.target not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.df[self.target], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{self.region} Distribution of {self.target}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Load MW")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3, axis='y')
        self._save_fig(fig, "06_distribution")

    def target_boxplot(self):
        if self.target not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(self.df[self.target], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_title(f"{self.region} Boxplot of {self.target}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Load MW")
        ax.grid(True, alpha=0.3, axis='y')
        self._save_fig(fig, "07_boxplot")

    def weekday_weekend(self):
        if "is_weekend" not in self.df.columns or self.target not in self.df.columns:
            return

        df = self.df.copy()
        df["weekend_label"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [df[df["weekend_label"] == "Weekday"][self.target].dropna(),
                df[df["weekend_label"] == "Weekend"][self.target].dropna()]
        bp = ax.boxplot(data, labels=["Weekday", "Weekend"], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_title(f"{self.region} Weekday versus Weekend Load", fontsize=14, fontweight='bold')
        ax.set_ylabel("Load MW")
        ax.grid(True, alpha=0.3, axis='y')
        self._save_fig(fig, "08_weekday_weekend")

    def temp_humidity_wind_scatter(self):
        for col in ["temperature", "humidity", "wind_speed"]:
            if col in self.df.columns and self.target in self.df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(self.df[col], self.df[self.target], alpha=0.3, s=10)
                ax.set_title(f"{self.region} {self.target} versus {col}", fontsize=14, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel("Load MW")
                ax.grid(True, alpha=0.3)
                self._save_fig(fig, f"09_scatter_{col}")

    def correlation_matrix(self):
        numeric = self.df.select_dtypes(include=["float32", "float64", "int32", "int64"])
        if numeric.shape[1] < 2:
            return

        corr = numeric.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, ha='right')
        ax.set_yticklabels(corr.columns)
        
        ax.set_title(f"{self.region} Correlation Matrix", fontsize=14, fontweight='bold')
        self._save_fig(fig, "10_correlation_matrix")

    def target_correlations_bar(self):
        numeric = self.df.select_dtypes(include=["float32", "float64", "int32", "int64"])
        if numeric.shape[1] < 2 or self.target not in numeric.columns:
            return

        corr = numeric.corr()[self.target].drop(self.target, errors="ignore")
        corr = corr.sort_values(ascending=False).head(10)
        
        if corr.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in corr.values]
        ax.barh(range(len(corr)), corr.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index)
        ax.set_xlabel("Correlation")
        ax.set_title(f"{self.region} Top Correlations with {self.target}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linewidth=0.8)
        self._save_fig(fig, "11_target_correlations")


# ============================================================================
# DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """
    Handles type conversions, target selection, time features,
    missing values, and sequence construction.
    """

    def __init__(self, seq_len=12):
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.target_col = None

    def prepare_dataframe(self, df):
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        for col in df.columns:
            if col == "timestamp":
                continue
            if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notna().sum() > 0:
                    df[col] = num

        for candidate in ["load_mw", "load_scaled", "value"]:
            if candidate in df.columns:
                self.target_col = candidate
                break

        if self.target_col is None:
            raise ValueError("No suitable target column found")

        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df[self.target_col] = df[self.target_col].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["dayofweek"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month

            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
            df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
            df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

        numeric_cols = df.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns
        feature_cols = [c for c in numeric_cols if c != self.target_col]

        df[feature_cols] = df[feature_cols].fillna(0.0)

        print(f"\nData preparation for target: {self.target_col}")
        print(f"Feature columns:", len(feature_cols))

        return df, self.target_col, feature_cols

    def build_sequences(self, df, target_col, feature_cols):
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.float32)

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        L = self.seq_len
        X_seq = []
        y_seq = []

        for i in range(len(df) - L):
            X_seq.append(X_raw[i:i + L])
            y_seq.append(y_raw[i + L])

        if not X_seq:
            print("Not enough rows to build sequences")
            return np.array([]), np.array([])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        print("Sequence shapes X", X_seq.shape, "y", y_seq.shape)
        return X_seq, y_seq


# ============================================================================
# H2O AutoML TRAINER
# ============================================================================

class H2OTrainer:
    """
    Trains H2O AutoML model
    """
    
    def __init__(self, region_name="Region"):
        self.region = region_name
        self.model = None
        self.results = {}
        self.h2o_initialized = False
        
    def init_h2o(self):
        if not H2O_AVAILABLE:
            print("H2O not available, skipping H2O AutoML")
            return False
            
        try:
            h2o.init(max_mem_size="4G")
            self.h2o_initialized = True
            print("H2O cluster initialized")
            return True
        except Exception as e:
            print("Could not initialize H2O", e)
            return False
    
    def train(self, X_train, X_test, y_train, y_test, feature_names):
        if not self.h2o_initialized:
            if not self.init_h2o():
                return None
        
        print("\n" + "=" * 60)
        print(f"Training H2O AutoML for {self.region}")
        print("=" * 60)
        
        try:
            if X_train.ndim == 3:
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                X_train_flat = X_train
                X_test_flat = X_test
            
            if X_train.ndim == 3:
                flat_feature_names = []
                for i in range(X_train.shape[1]):
                    for fname in feature_names:
                        flat_feature_names.append(f"{fname}_t{i}")
            else:
                flat_feature_names = feature_names
            
            train_df = pd.DataFrame(X_train_flat, columns=flat_feature_names)
            train_df["target"] = y_train
            
            test_df = pd.DataFrame(X_test_flat, columns=flat_feature_names)
            test_df["target"] = y_test
            
            train_h2o = h2o.H2OFrame(train_df)
            test_h2o = h2o.H2OFrame(test_df)
            
            train_h2o["target"] = train_h2o["target"].asnumeric()
            test_h2o["target"] = test_h2o["target"].asnumeric()
            
            predictors = flat_feature_names
            response = "target"
            
            print(f"Starting H2O AutoML {H2O_MAX_RUNTIME_SECS} seconds max")
            aml = H2OAutoML(
                max_runtime_secs=H2O_MAX_RUNTIME_SECS,
                max_models=H2O_MAX_MODELS,
                seed=42,
                sort_metric="RMSE"
            )
            
            aml.train(x=predictors, y=response, training_frame=train_h2o)
            
            preds_h2o = aml.leader.predict(test_h2o)
            predictions = h2o.as_list(preds_h2o)["predict"].values
            
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))
            
            self.model = aml
            self.results = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pred": predictions,
                "leaderboard": aml.leaderboard.as_data_frame()
            }
            
            print("\nH2O AutoML Complete")
            print("Best Model:", aml.leader.model_id)
            print(f"RMSE {rmse:.4f}")
            print(f"MAE {mae:.4f}")
            print(f"R2 {r2:.4f}")
            
            print("\nTop models leaderboard")
            print(aml.leaderboard.head(5))
            
            return self.results
            
        except Exception as e:
            print("Error training H2O AutoML", e)
            import traceback
            traceback.print_exc()
            return None
    
    def save_model(self):
        if self.model is None:
            return
        
        try:
            model_path = os.path.join(Config.MODELS, f"{self.region}_H2O_AutoML")
            h2o.save_model(model=self.model.leader, path=model_path, force=True)
            print("Saved H2O model", model_path)
        except Exception as e:
            print("Error saving H2O model", e)


# ============================================================================
# MODEL TRAINER
# ============================================================================

class MLTrainer:
    """
    Trains models on flattened sequences.
    """

    def __init__(self, region_name="Region"):
        self.region = region_name
        self.results = {}
        self.best_model_name = None

    def train_all(self, X_train, X_test, y_train, y_test):
        if X_train.ndim == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test

        X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_flat = np.nan_to_num(X_test_flat, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"\nModel training for region: {self.region}")
        print("X_train_flat", X_train_flat.shape, "X_test_flat", X_test_flat.shape)

        models = {}

        models["LinearRegression"] = LinearRegression(n_jobs=-1)
        models["Ridge"] = Ridge(alpha=1.0)
        models["Lasso"] = Lasso(alpha=0.001, max_iter=500)

        models["RandomForest"] = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        models["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        if xgb is not None:
            xgb_extra = {}
            if USE_GPU:
                xgb_extra = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
            else:
                xgb_extra = {"tree_method": "hist"}

            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **xgb_extra,
            )

        if lgb is not None:
            lgb_extra = {}
            if USE_GPU:
                lgb_extra = {"device_type": "gpu"}

            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=80,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **lgb_extra,
            )

        for name, model in models.items():
            print("\nTraining model:", name)
            try:
                model.fit(X_train_flat, y_train)
                pred = model.predict(X_test_flat)

                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                mae = float(mean_absolute_error(y_test, pred))
                r2 = float(r2_score(y_test, pred))

                self.results[name] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "pred": pred,
                    "model": model,
                }

                print(f"{name} RMSE {rmse:.4f} MAE {mae:.4f} R2 {r2:.4f}")

            except Exception as e:
                print("Error training", name, e)

        if self.results:
            self.best_model_name = builtins.min(
                self.results.keys(), key=lambda k: self.results[k]["rmse"]
            )
            best = self.results[self.best_model_name]
            print(
                f"\nBest model for {self.region}: {self.best_model_name} "
                f"RMSE {best['rmse']:.4f} R2 {best['r2']:.4f}"
            )

    def save_models(self):
        count = 0
        for name, res in self.results.items():
            model = res["model"]
            safe_name = f"{self.region}_{name}".replace(" ", "_")
            path = os.path.join(Config.MODELS, f"{safe_name}.pkl")
            try:
                joblib.dump(model, path)
                count += 1
                print("Saved model", safe_name)
            except Exception as e:
                print("Error saving model", name, e)
        print("Total models saved for", self.region, count)

    def plot_comparison(self):
        if not self.results:
            return

        models = list(self.results.keys())
        rmse = [self.results[m]["rmse"] for m in models]
        r2 = [self.results[m]["r2"] for m in models]
        mae = [self.results[m]["mae"] for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].bar(models, rmse, color='steelblue', alpha=0.7)
        axes[0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rmse):
            axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        axes[1].bar(models, r2, color='green', alpha=0.7)
        axes[1].set_title('R2 Score Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('R2')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(r2):
            axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        axes[2].bar(models, mae, color='coral', alpha=0.7)
        axes[2].set_title('MAE Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae):
            axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        fig.suptitle(f'{self.region} Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        img_path = os.path.join(Config.IMAGES, f"{self.region}_model_comparison.png")
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        print("Saved model comparison", img_path)
        plt.close(fig)

    def plot_best_predictions(self, y_test):
        if not self.best_model_name:
            return

        name = self.best_model_name
        pred = self.results[name]["pred"]
        n = safe_min(200, len(y_test), len(pred))

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(y_test[:n], label='Actual', linewidth=1.5, alpha=0.7)
        axes[0].plot(pred[:n], label='Predicted', linewidth=1.5, alpha=0.7)
        axes[0].set_title(f'{self.region} {name} Predictions', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Load MW')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_test[:n], pred[:n], alpha=0.5, s=20)
        axes[1].plot([y_test[:n].min(), y_test[:n].max()], 
                     [y_test[:n].min(), y_test[:n].max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title('Actual versus Predicted', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Actual Load MW')
        axes[1].set_ylabel('Predicted Load MW')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_name = f"{self.region}_{name}".replace(" ", "_")
        img_path = os.path.join(Config.IMAGES, f"{safe_name}_predictions.png")
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        print("Saved predictions", img_path)
        plt.close(fig)


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    @staticmethod
    def build_report_dict(region, df, target_col, feature_cols, trainer, h2o_trainer, total_sequences):
        report = {
            "project_info": {
                "title": "Electricity Load Forecasting with H2O AutoML",
                "region": region,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "team": [
                    "Chaitanya Datta Maddukuri",
                    "Meghana Rabba",
                    "Rohan Singh Rajendra Singh",
                    "Tanushree Sharma",
                ],
            },
            "data_summary": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "target_column": target_col,
                "feature_columns": int(len(feature_cols)),
                "sequences_created": int(total_sequences),
            },
            "model_performance": {},
            "best_model": None,
        }

        if trainer.results:
            for name, res in trainer.results.items():
                report["model_performance"][name] = {
                    "rmse": float(res["rmse"]),
                    "mae": float(res["mae"]),
                    "r2": float(res["r2"]),
                }

        if h2o_trainer and h2o_trainer.results:
            report["model_performance"]["H2O_AutoML"] = {
                "rmse": float(h2o_trainer.results["rmse"]),
                "mae": float(h2o_trainer.results["mae"]),
                "r2": float(h2o_trainer.results["r2"]),
            }

        if report["model_performance"]:
            best_name = builtins.min(
                report["model_performance"].keys(),
                key=lambda k: report["model_performance"][k]["rmse"]
            )
            best_perf = report["model_performance"][best_name]
            report["best_model"] = {
                "name": best_name,
                "rmse": float(best_perf["rmse"]),
                "mae": float(best_perf["mae"]),
                "r2": float(best_perf["r2"]),
            }

        return report

    @staticmethod
    def save_json(report, region):
        path = os.path.join(Config.REPORTS, f"{region}_model_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print("Saved JSON report", path)

    @staticmethod
    def save_html(report, region):
        path = os.path.join(Config.REPORTS, f"{region}_project_report.html")
        best = report.get("best_model")

        html = []
        html.append("<html><head>")
        html.append("<title>CSP 554 Load Forecasting Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }")
        html.append(".container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }")
        html.append("h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
        html.append("h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }")
        html.append("h3 { color: #7f8c8d; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        html.append("th { background-color: #3498db; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("tr:hover { background-color: #e8f4f8; }")
        html.append(".best { background-color: #2ecc71 !important; color: white; font-weight: bold; }")
        html.append(".metric-box { background: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0; }")
        html.append("ul { line-height: 1.8; }")
        html.append(".info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }")
        html.append(".info-item { background: #f8f9fa; padding: 15px; border-radius: 5px; }")
        html.append("</style>")
        html.append("</head><body>")
        html.append("<div class='container'>")
        
        html.append(f"<h1>Electricity Load Forecasting {region}</h1>")
        html.append(f"<p style='color: #7f8c8d; font-style: italic;'>Generated: {report['project_info']['date']}</p>")

        html.append("<h2>Project Summary</h2>")
        html.append("<h3>Team Members</h3><ul>")
        for m in report["project_info"]["team"]:
            html.append(f"<li>{m}</li>")
        html.append("</ul>")

        ds = report["data_summary"]
        html.append("<h3>Data Summary</h3>")
        html.append("<div class='info-grid'>")
        html.append(f"<div class='info-item'><strong>Total Rows:</strong> {ds['rows']:,}</div>")
        html.append(f"<div class='info-item'><strong>Total Columns:</strong> {ds['columns']}</div>")
        html.append(f"<div class='info-item'><strong>Target Column:</strong> {ds['target_column']}</div>")
        html.append(f"<div class='info-item'><strong>Feature Columns:</strong> {ds['feature_columns']}</div>")
        html.append(f"<div class='info-item'><strong>Sequences Created:</strong> {ds['sequences_created']:,}</div>")
        html.append("</div>")

        if best:
            html.append("<h2>Best Model</h2>")
            html.append("<div class='metric-box'>")
            html.append(f"<h3 style='margin-top: 0; color: #2ecc71;'>{best['name']}</h3>")
            html.append(f"<p><strong>RMSE:</strong> {best['rmse']:.4f}</p>")
            html.append(f"<p><strong>R2 Score:</strong> {best['r2']:.4f}</p>")
            html.append(f"<p><strong>MAE:</strong> {best['mae']:.4f}</p>")
            html.append("</div>")

        html.append("<h2>All Model Performance</h2>")
        html.append("<table>")
        html.append("<tr><th>Model</th><th>RMSE</th><th>R2 Score</th><th>MAE</th></tr>")
        
        for name, mp in sorted(report["model_performance"].items(), 
                              key=lambda x: x[1]["rmse"]):
            row_class = " class='best'" if (best and name == best['name']) else ""
            html.append(
                f"<tr{row_class}><td>{name}</td>"
                f"<td>{mp['rmse']:.4f}</td>"
                f"<td>{mp['r2']:.4f}</td>"
                f"<td>{mp['mae']:.4f}</td></tr>"
            )
        html.append("</table>")

        html.append("<hr style='margin: 40px 0; border: none; border-top: 2px solid #ddd;'>")
        html.append("<p style='text-align: center; color: #95a5a6;'><em>Report generated by CSP 554 ML Pipeline with H2O AutoML</em></p>")
        html.append("</div></body></html>")

        with open(path, "w", encoding="utf8") as f:
            f.write("\n".join(html))

        print("Saved HTML report", path)


# ============================================================================
# REGION LOADERS
# ============================================================================

def load_region(region_name):
    """
    Loads every parquet file in a region folder from S3.
    Folder format: region=name
    """
    region_path = f"{BASE_PATH}/region={region_name}"
    print(f"\nLoading region dataset {region_name} from {region_path}")
    dataset = pq.ParquetDataset(region_path)
    df = dataset.read().to_pandas()
    print(f"{region_name} loaded shape {df.shape}")
    return df


def load_all_regions():
    """
    Load known regions from S3.
    """
    regions = {}
    region_names = ["AEP", "COMED", "PJM", "DOM"]

    for region_name in region_names:
        try:
            df = load_region(region_name)
            regions[region_name] = df
        except FileNotFoundError as e:
            print("Region missing", region_name, e)
        except Exception as e:
            print("Error loading region", region_name, e)

    print("\nLoaded regions:", list(regions.keys()))
    return regions


# ============================================================================
# COMPLETE PIPELINE FOR ONE REGION
# ============================================================================

def run_pipeline_from_dataframe(df, region_name="Region"):
    print("\n" + "=" * 80)
    print(f"COMPLETE DATA SCIENCE PIPELINE FOR REGION {region_name}")
    print("=" * 80)
    print("Initial shape", df.shape)

    pre = DataPreprocessor(seq_len=12)
    df_clean, target_col, feature_cols = pre.prepare_dataframe(df)

    eda = MatplotlibEDAAnalyzer(df_clean, target_col=target_col, time_col="timestamp", region_name=region_name)
    eda.run_all()

    X_seq, y_seq = pre.build_sequences(df_clean, target_col, feature_cols)
    if X_seq.size == 0:
        print("No sequences created, stopping pipeline for", region_name)
        return None

    total_sequences = len(X_seq)
    split = int(0.8 * total_sequences)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print("\nTrain sizes X", X_train.shape, "y", y_train.shape)
    print("Test sizes X", X_test.shape, "y", y_test.shape)

    trainer = MLTrainer(region_name=region_name)
    trainer.train_all(X_train, X_test, y_train, y_test)
    trainer.plot_comparison()
    trainer.plot_best_predictions(y_test)
    trainer.save_models()

    h2o_trainer = H2OTrainer(region_name=region_name)
    h2o_results = h2o_trainer.train(X_train, X_test, y_train, y_test, feature_cols)
    
    if h2o_results:
        n = safe_min(200, len(y_test))
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(y_test[:n], label='Actual', linewidth=1.5, alpha=0.7)
        axes[0].plot(h2o_results['pred'][:n], label='H2O Predicted', linewidth=1.5, alpha=0.7)
        axes[0].set_title(f'{region_name} H2O AutoML Predictions', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Load MW')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_test[:n], h2o_results['pred'][:n], alpha=0.5, s=20)
        axes[1].plot([y_test[:n].min(), y_test[:n].max()], 
                     [y_test[:n].min(), y_test[:n].max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title('Actual versus H2O Predicted', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Actual Load MW')
        axes[1].set_ylabel('Predicted Load MW')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_path = os.path.join(Config.IMAGES, f"{region_name}_H2O_AutoML_predictions.png")
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        print("Saved H2O predictions", img_path)
        plt.close(fig)
        
        h2o_trainer.save_model()

    report = ReportGenerator.build_report_dict(
        region_name,
        df_clean,
        target_col,
        feature_cols,
        trainer,
        h2o_trainer,
        total_sequences,
    )
    ReportGenerator.save_json(report, region_name)
    ReportGenerator.save_html(report, region_name)

    print("\n" + "=" * 80)
    print("Pipeline finished for region", region_name)
    print("Traditional ML models trained", len(trainer.results))
    print("Best traditional model", trainer.best_model_name)
    if h2o_results:
        print("H2O AutoML trained")
        print(f"H2O RMSE {h2o_results['rmse']:.4f} R2 {h2o_results['r2']:.4f}")
    print("=" * 80 + "\n")
    
    return {
        "trainer": trainer,
        "h2o_trainer": h2o_trainer
    }


# ============================================================================
# RUN PIPELINE FOR EVERY REGION
# ============================================================================

def run_pipeline_for_all_regions():
    regions = load_all_regions()
    all_trainers = {}

    for region_name, df in regions.items():
        print("\n" + "=" * 100)
        print(f"RUNNING PIPELINE FOR REGION {region_name}")
        print("=" * 100)
        
        result = run_pipeline_from_dataframe(df, region_name=region_name)
        all_trainers[region_name] = result

    if H2O_AVAILABLE:
        try:
            h2o.cluster().shutdown(prompt=False)
            print("\nH2O cluster shut down")
        except Exception:
            pass

    print("\n" + "=" * 100)
    print("ALL PIPELINES COMPLETED")
    print("=" * 100)
    print("Processed regions", list(all_trainers.keys()))
    print("Outputs saved to", Config.BASE)
    print("Models folder", Config.MODELS)
    print("Reports folder", Config.REPORTS)
    print("Images folder", Config.IMAGES)
    print("=" * 100)
    
    return all_trainers


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("ELECTRICITY LOAD FORECASTING PIPELINE WITH H2O AUTOML")
    print("=" * 100)
    print("H2O Available:", H2O_AVAILABLE)
    print("XGBoost Available:", xgb is not None)
    print("LightGBM Available:", lgb is not None)
    print("Using Matplotlib for visualizations")
    print("=" * 100)
    
    all_trainers = run_pipeline_for_all_regions()

