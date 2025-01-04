import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql import functions as F
import json
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, sum
from math import pi, cos, sin

# Path to Parquet file
flight_parquet_path = './data/flights.parquet'
planes_parquet_path = './data/planes.parquet'
# Path to schema file
plane_schema_file_path = './data/plane-schema.json'
flight_schema_file_path = './data/flight-schema.json'
# Load paths
flight_wildcard_path = './data/*.csv.bz2'
plane_path = './data/plane-data.csv'
# airport_path = './data/plane-data.csv'
# Result paths
processed_train_path = "./data/processed/train.pkl"
processed_test_path = "./data/processed/test.pkl"
processed_schema_path = "./data/processed/schema.pkl"


def load_csv_save_parquet(spark, raw_path, parquet_path, schema_path) -> DataFrame:
    # Read csv
    df = spark.read.csv(
        raw_path,
        header=True,
        inferSchema=True
    )

    schema = df.schema
    schema_json = schema.json()

    # Write the schema JSON to a file
    with open(schema_path, 'w') as f:
        f.write(schema_json)

    # Save DataFrame as Parquet for future use
    df.repartition(1)
    df.write.parquet(parquet_path)

    df = spark.read.parquet(parquet_path, schema=schema)
    return df


def load_parquet(spark, parquet_path, schema_file) -> DataFrame:
    with open(schema_file, 'r') as f:
        schema_json = f.read()

    # Deserialize the JSON string back into a StructType object
    schema_from_file = StructType.fromJson(json.loads(schema_json))

    df = spark.read.parquet(parquet_path, schema=schema_from_file)
    return df


def load(spark, parquet_path, schema_file_path, wildcard_path) -> (DataFrame, DataFrame):
    if os.path.exists(parquet_path):
        # If Parquet exists, load it using the schema files
        df = load_parquet(spark, parquet_path, schema_file_path)
        df_planes = load_parquet(spark, planes_parquet_path, plane_schema_file_path)
    else:
        # If Parquet file does not exist, read CSV files and save as Parquet
        df = load_csv_save_parquet(spark, wildcard_path, parquet_path, schema_file_path)
        df_planes = load_csv_save_parquet(spark, plane_path, planes_parquet_path, plane_schema_file_path)
    return df, df_planes


def custom_polar_time_encode(df):
    print(f"Transforming Month, DayofMonth and DayofWeek to polar coordinates.")

    def polar_encoding(value, max_value):
        frac = value / max_value
        circle = 2 * pi
        angle = frac * circle
        return cos(angle), sin(angle)

    # Register UDF for polar encoding
    polar_udf = udf(polar_encoding, "struct<cos:double, sin:double>")

    # Apply polar encoding on 'Month', 'DayofMonth', 'DayOfWeek'
    df = df.withColumn("Month_polar", polar_udf(col("Month"), lit(12))) \
        .withColumn("DayofMonth_polar", polar_udf(col("DayofMonth"),
                                                  when(col("Month") == 2, lit(28))  # February (can adjust for leap year
                                                  # if needed)
                                                  .when(col("Month").isin([4, 6, 9, 11]),
                                                        lit(30))  # Months with 30 days
                                                  .otherwise(lit(31)))) \
        .withColumn("DayOfWeek_polar", polar_udf(col("DayOfWeek"), lit(7)))
    # df = df.drop(*["DayofMonth", "DayOfWeek", "Month"])

    # Subdivide feature pairs into two columns each
    df = df.withColumn("Month_cos", col("Month_polar.cos")) \
        .withColumn("Month_sin", col("Month_polar.sin")) \
        .withColumn("DayofMonth_cos", col("DayofMonth_polar.cos")) \
        .withColumn("DayofMonth_sin", col("DayofMonth_polar.sin")) \
        .withColumn("DayOfWeek_cos", col("DayOfWeek_polar.cos")) \
        .withColumn("DayOfWeek_sin", col("DayOfWeek_polar.sin"))

    df = df.drop(*["DayofMonth_polar", "DayOfWeek_polar", "Month_polar"])

    return df


def static_preprocess(df):
    print("Schema before static preprocessing")
    df.printSchema()

    forbidden_cols = [
        "ArrTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "Diverted",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay"
    ]
    df = df.drop(*forbidden_cols)

    target_column = "ArrDelay"

    # List of Ordinal features
    cyclic_ordinal_time = [
        'Month',
        'DayofMonth',
        'DayOfWeek'
    ]
    non_cyclic_ordinal_time = ['Year', 'PlaneIssueYear']

    # List of Time features
    quant_time_features = [
        'DepTime',
        'CRSDepTime',
        'CRSArrTime'
    ]

    # List of Quantitative features
    quantitative_features = [
        'CRSElapsedTime',
        'DepDelay',
        'Distance',
        'TaxiOut'
    ]

    # List of Nominal features
    nominal_features = [
        'UniqueCarrier',
        'FlightNum',
        'TailNum',
        'Origin',
        'Dest',
        'Cancelled',
        'CancellationCode',
        'EngineType',
        'AircraftType',
        'Manufacturer',
        'Model',
        "issue_date", "status",
        "type"
    ]

    # WE ARE PREDICTING DELAY. REMOVE CANCELLED FLIGHTS
    df = df.filter("Cancelled != 1")
    df = df.dropna(subset=[target_column])

    # DROP NOMINALS WITH TOO MANY GROUPS OR THAT ARE USELESS
    useless_fea = ["TailNum", "FlightNum", "UniqueCarrier", "CancellationCode", "Cancelled", "issue_date", "status",
                   "type"]
    for fea in useless_fea:
        print(f"Discarding {fea}.")
        nominal_features.remove(fea)
    df = df.drop(*useless_fea)

    # RENAME VARIABLES
    df = df.withColumnRenamed("year", "PlaneIssueYear")
    df = df.withColumnRenamed("engine_type", "EngineType")
    df = df.withColumnRenamed("aircraft_type", "AircraftType")
    df = df.withColumnRenamed("model", "Model")
    df = df.withColumnRenamed("manufacturer", "Manufacturer")

    # CAST QUANTITATIVE COLUMNS TO NUMERIC, SOME ARE STRINGS
    for column in quantitative_features + [target_column]:
        print(f"Forcing {column} to be read as integer.")
        df = df.withColumn(column, col(column).cast(IntegerType()))

    # CAST HHMM COLUMNS TO MINUTE QUANTITIES
    for column in quant_time_features:  # They are strings hhmm
        print(f"Casting {column} from hhmm to minutes (integer).")
        df = df.withColumn(
            column + "_minutes",
            (F.col(column).substr(1, 2).cast("int") * 60 + F.col(column).substr(3, 2).cast("int"))
        )
        quantitative_features.append(column + "_minutes")
    df = df.drop(*quant_time_features)

    df = custom_polar_time_encode(df)
    ordinal_features = []
    ordinal_features += [fea + "_sin" for fea in cyclic_ordinal_time]
    ordinal_features += [fea + "_cos" for fea in cyclic_ordinal_time]

    return df, quantitative_features, ordinal_features, nominal_features + non_cyclic_ordinal_time


def preprocess_nominal(train_df, test_df, nominal_features):
    for fea in nominal_features:
        if fea in ["Origin", "Dest"]:
            continue

        print(f"Performing One-Hot-Encoding to feature {fea}")

        indexer = StringIndexer(inputCol=f"{fea}", outputCol=f"{fea}_index", handleInvalid='keep')
        encoder = OneHotEncoder(inputCol=f"{fea}_index", outputCol=f"{fea}_binary", handleInvalid='keep',
                                dropLast=True)

        pipeline = Pipeline(stages=[indexer, encoder])
        year_pipeline = pipeline.fit(train_df)
        train_df = year_pipeline.transform(train_df)
        test_df = year_pipeline.transform(test_df)
        train_df = train_df.drop(f"{fea}_index")
        test_df = test_df.drop(f"{fea}_index")
        print(test_df.head())
        print(train_df.head())

    print(f"Performing One-Hot-Encoding to joint features Origin and Dest")
    origins = train_df.select("Origin")

    destinations = train_df.select("Dest")
    destinations = destinations.withColumnRenamed("Dest", "Origin")
    stacked_df = origins.union(destinations)
    indexer = StringIndexer(inputCol="Origin", outputCol=f"Origin_index", handleInvalid='keep').fit(stacked_df)
    encoder = OneHotEncoder(inputCol=f"Origin_index", outputCol=f"Origin_binary", handleInvalid='keep',
                            dropLast=True).fit(indexer.transform(stacked_df))

    train_df = encoder.transform(indexer.transform(train_df))
    test_df = encoder.transform(indexer.transform(test_df))
    train_df = train_df.drop(f"Origin_index")
    test_df = test_df.drop(f"Origin_index")

    indexer.setInputCol("Dest").setOutputCol("Dest_index")
    encoder.setInputCol("Dest_index").setOutputCol("Dest_binary")
    train_df = encoder.transform(indexer.transform(train_df))
    test_df = encoder.transform(indexer.transform(test_df))

    train_df = train_df.drop(f"Dest_index")
    test_df = test_df.drop(f"Dest_index")

    print(test_df.head())
    print(train_df.head())
    return train_df, test_df


def preprocess(flight_df, df_planes, train_frac):
    df_planes = df_planes.withColumnRenamed("tailnum", "TailNum")
    flight_df = flight_df.join(df_planes, on="TailNum", how="inner")

    flight_df, quantitative_features, ordinal_features, nominal_features = static_preprocess(flight_df)

    train_df, test_df = flight_df.randomSplit([train_frac, 1 - train_frac], seed=42)

    print("Current schema")
    train_df.printSchema()

    for fea in nominal_features + ordinal_features:
        if flight_df.filter(col(fea).isNull()).count() > 0:
            print(f"Filling {fea} with mode")
            mode = train_df.groupby(fea).count().orderBy("count", ascending=False).first()[0]
            train_df = train_df.fillna(mode, subset=fea)
            test_df = test_df.fillna(mode, subset=fea)

    for fea in quantitative_features:
        if flight_df.filter(col(fea).isNull()).count() > 0:
            print(f"Filling {fea} with median")
            median = train_df.approxQuantile(col=fea, probabilities=[0.5], relativeError=0.05)[0]
            train_df = train_df.fillna(median, subset=fea)
            test_df = test_df.fillna(median, subset=fea)

    # Calculate null count for each column
    print("Printing null counts")
    null_counts = train_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in
                                   quantitative_features + nominal_features + ordinal_features])
    null_counts.show()

    train_df, test_df = preprocess_nominal(train_df, test_df, nominal_features)

    # Quantitative feature assembly
    quant_assembler = VectorAssembler(
        inputCols=quantitative_features,
        outputCol="quant_features_vector"
    )

    # Assemble encoded nominal features
    nominal_assembler = VectorAssembler(
        inputCols=[f"{col}_binary" for col in nominal_features],
        outputCol="nominal_features_vector"
    )

    ordinal_assembler = VectorAssembler(
        inputCols=[col for col in ordinal_features],
        outputCol="ordinal_features_vector"
    )

    # Final feature vector
    final_assembler = VectorAssembler(
        inputCols=["quant_features_vector", "nominal_features_vector", "ordinal_features_vector"],
        outputCol="features"
    )

    # Create a pipeline
    pipeline = Pipeline(stages=[ordinal_assembler,
                                quant_assembler,
                                nominal_assembler,
                                final_assembler
                                ])

    # Fit and transform the dataframe
    pipeline_model = pipeline.fit(train_df)
    train_df = pipeline_model.transform(train_df)
    test_df = pipeline_model.transform(test_df)

    print("With vectorization of all features")
    train_df.printSchema()
    print(train_df.head())
    print(test_df.head())

    return train_df, test_df


def main(n_partitions=50):
    spark = (SparkSession.builder.appName("MachineLearningProject")
             .config("spark.executor.memory", "4g")
             .config("spark.driver.memory", "48g")
             .config("spark.memory.fraction", "0.8")
             .config("spark.memory.storageFraction", "0.3")
             .config("spark.driver.maxResultSize", "4g")
             .config("spark.sql.caseSensitive", "true")
             .config("spark.local.dir", "./temp/")
             .getOrCreate())

    if not os.path.exists(processed_train_path):
        df, df_planes = load(spark, flight_parquet_path, plane_schema_file_path, flight_wildcard_path)
        df = df.repartition(n_partitions)
        train_df, test_df = preprocess(df, df_planes, train_frac=0.8)
        print("Finished preprocessing")
        print(train_df.head())
        print(test_df.head())

        with open(processed_test_path, 'wb') as f:
            pickle.dump(test_df.collect(), f)
        with open(processed_train_path, 'wb') as f:
            pickle.dump(train_df.collect(), f)
        with open(processed_schema_path, 'wb') as f:
            pickle.dump(train_df.schema, f)
    else:
        with open(processed_schema_path, 'wb') as f:
            schema = pickle.load(f)
        with open(processed_train_path, 'rb') as f:
            train_df = pickle.load(f)
        with open(processed_test_path, 'rb') as f:
            test_df = pickle.load(f)

        train_df = spark.createDataFrame(train_df, schema)
        test_df = spark.createDataFrame(test_df, schema)

    print(train_df.head())
    spark.stop()


if __name__ == "__main__":
    import os

    os.environ['PYSPARK_PYTHON'] = r'C:\Users\franb\AppData\Local\Programs\Python\Python38\python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\franb\AppData\Local\Programs\Python\Python38\python.exe'
    # os.environ["PATH"] = r"C:\Users\franb\AppData\Local\Programs\Python\Python38" + os.pathsep + os.environ["PATH"]

    main()
