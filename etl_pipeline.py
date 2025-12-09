from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_timestamp, to_date, dayofweek, when,
    min as spark_min, max as spark_max,
    mean as spark_mean, stddev, explode, sequence, year, month, avg, lag
)
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("electricity-etl").getOrCreate()

BUCKET = "electricity-forcast-tanushree"

ELECTRICITY_BASE = f"s3://{BUCKET}/Electricity /"
WEATHER_BASE     = f"s3://{BUCKET}/Weather/"
HOLIDAY_BASE     = f"s3://{BUCKET}/Holiday/"

OUTPUT_PATH      = f"s3://{BUCKET}/processed/features/"


def read_region(path, region_name):
    """
    Generic reader:
    - assumes first column is datetime, second is load
    - renames second column to 'load_mw'
    - adds 'region' column
    """
    df = spark.read.option("header", True).csv(path)
    cols = df.columns
    if len(cols) < 2:
        raise ValueError(f"{region_name}: expected at least 2 columns, got {len(cols)}")

    datetime_col = cols[0]
    load_col = cols[1]

    df = (
        df
        .withColumn("region", lit(region_name))
        .withColumnRenamed(load_col, "load_mw")
        .withColumnRenamed(datetime_col, "Datetime")
    )
    return df

dom    = read_region(ELECTRICITY_BASE + "DOM_hourly.csv",        "DOM")
comed  = read_region(ELECTRICITY_BASE + "COMED_hourly.csv",      "COMED")
aep    = read_region(ELECTRICITY_BASE + "AEP_hourly.csv",        "AEP")
dayton = read_region(ELECTRICITY_BASE + "DAYTON_hourly.csv",     "DAYTON")
pjm    = read_region(ELECTRICITY_BASE + "PJM_Load_hourly.csv",   "PJM")

electricity = (
    dom.unionByName(comed)
       .unionByName(aep)
       .unionByName(dayton)
       .unionByName(pjm)
)

electricity = electricity.withColumn(
    "timestamp", to_timestamp(col("Datetime"))
).drop("Datetime")

temperature = spark.read.option("header", True).csv(WEATHER_BASE + "temperature.csv")
humidity    = spark.read.option("header", True).csv(WEATHER_BASE + "humidity.csv")
wind_speed  = spark.read.option("header", True).csv(WEATHER_BASE + "wind_speed.csv")

if "datetime" in temperature.columns:
    temperature = temperature.withColumn("timestamp", to_timestamp(col("datetime"))).drop("datetime")
if "datetime" in humidity.columns:
    humidity = humidity.withColumn("timestamp", to_timestamp(col("datetime"))).drop("datetime")
if "datetime" in wind_speed.columns:
    wind_speed = wind_speed.withColumn("timestamp", to_timestamp(col("datetime"))).drop("datetime")


# Replace file name if your holiday CSV is named differently
holidays = spark.read.option("header", True).csv(HOLIDAY_BASE + "United States_US.csv")

date_col = None
for cand in ["Date", "date", "ds"]:
    if cand in holidays.columns:
        date_col = cand
        break

if date_col is None:
    raise ValueError("Cannot find a date column in holiday CSV. Expected one of: Date/date/ds")

holidays = (
    holidays
    .withColumn("date", to_date(col(date_col)))
    .withColumn("is_holiday", lit(1))
    .select("date", "is_holiday")
)

# Builiding full hourly grid for electricity

bounds = electricity.agg(
    spark_min("timestamp").alias("min_ts"),
    spark_max("timestamp").alias("max_ts")
).collect()[0]

min_ts = bounds["min_ts"]
max_ts = bounds["max_ts"]

# Hourly timestamps between min and max
ts_df = spark.sql(f"""
    SELECT explode(
        sequence(
            timestamp('{min_ts}'),
            timestamp('{max_ts}'),
            interval 1 hour
        )
    ) as timestamp
""")

regions_df = electricity.select("region").distinct()

# grid (region, timestamp)
grid = regions_df.crossJoin(ts_df)

elec_full = grid.join(
    electricity.select("region", "timestamp", "load_mw"),
    on=["region", "timestamp"],
    how="left"
).fillna({"load_mw": 0.0})

#  adding holiday /weekend features

df = elec_full.withColumn("date", to_date("timestamp"))

df = df.join(holidays, on="date", how="left").fillna({"is_holiday": 0})

df = df.withColumn("day_of_week", dayofweek("timestamp")) \
       .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), 1).otherwise(0))

# Lag and rolling feature

w = Window.partitionBy("region").orderBy("timestamp")

df = (
    df
    .withColumn("lag_1h",  lag("load_mw", 1).over(w))
    .withColumn("lag_24h", lag("load_mw", 24).over(w))
    .withColumn("lag_7d",  lag("load_mw", 24*7).over(w))
)

w24 = w.rowsBetween(-24, -1)
df = df.withColumn("rolling_mean_24h", avg("load_mw").over(w24))

# final 

df = df.withColumn("year", year("timestamp")) \
       .withColumn("month", month("timestamp"))

final_cols = [
    "timestamp", "region",
    "load_mw", "load_scaled",
    "is_holiday", "is_weekend",
    "lag_1h", "lag_24h", "lag_7d",
    "rolling_mean_24h",
    "year", "month"
]

final = df.select(*final_cols)

final.write.mode("overwrite") \
     .partitionBy("region", "year", "month") \
     .parquet(OUTPUT_PATH)

print(f"ETL complete. Output written to {OUTPUT_PATH}")
