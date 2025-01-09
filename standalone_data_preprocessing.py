import pickle
import shutil

from pyspark import keyword_only
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline, Transformer, Estimator, PipelineModel
from pyspark.ml.feature import *
from pyspark.sql import functions as F
import json
from pyspark.sql.types import IntegerType, DoubleType, FloatType, StructField, Row
from pyspark.sql.functions import col, sum
from math import pi, cos, sin
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.context import SparkContext as sc
import os

TARGET_COLUMN = "ArrDelay"
# Path to Parquet file
FLIGHT_PARQUET_PATH = './data/flights.parquet'
PLANES_PARQUET_PATH = './data/planes.parquet'
PROCESSING_DIR = "data/processing/"
# Path to schema file
PLANE_SCHEMA_PATH = './data/plane-schema.json'
FLIGHT_SCHEMA_PATH = './data/flight-schema.json'
# Load paths
FLIGHT_RAW_PATH = './data/*.csv.bz2'
PLANE_RAW_PATH = './data/plane-data.csv'
# Result paths
PROCESSED_DIR = './data/processed/'
PROCESSED_TRAIN_PARQUET = os.path.join(PROCESSED_DIR, "train.parquet")
PROCESSED_TEST_PARQUET = os.path.join(PROCESSED_DIR, "test.parquet")
PROCESSED_SCHEMA = os.path.join(PROCESSED_DIR, "schema.json")


def custom_mean_encoding(df, mapping, feature):
    df = df.join(mapping, how="left", on=feature)
    m = df.approxQuantile(col="ArrDelay", probabilities=[0.5], relativeError=0.001)[0]
    df = df.withColumn(
        f"{feature}_mean_enc", F.coalesce(F.col(f"{feature}_mean_enc"), F.lit(m))
    )
    return df


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
        df_planes = load_parquet(spark, PLANES_PARQUET_PATH, PLANE_SCHEMA_PATH)
    else:
        # If Parquet file does not exist, read CSV files and save as Parquet
        df = load_csv_save_parquet(spark, wildcard_path, parquet_path, schema_file_path)
        df_planes = load_csv_save_parquet(spark, PLANE_RAW_PATH, PLANES_PARQUET_PATH, PLANE_SCHEMA_PATH)
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


def static_preprocess(df, df_planes):
    df_planes = df_planes.withColumnRenamed("tailnum", "TailNum")
    df = df.join(df_planes, on="TailNum", how="inner")

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
    df = df.dropna(subset=[target_column])
    null_count = df.filter(col(target_column).isNull()).count()
    print(f"Number of nulls in {target_column}: {null_count}")

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


def train_preprocess(df, nominal_features, ordinal_features, quantitative_features, dir_save_params,
                     cardinality_threshold, frequency_threshold, high_cardinality_strategy):
    spark = SparkSession.builder.getOrCreate()
    # -------------------------------- IMPUTER --------------------------------
    # This should be the column, the values considered nulls, and the value to be used to fill

    print("Analyzing medians")
    imputer_maps = {
        fea: {'extra_nulls': [],
              'fill_value': df.approxQuantile(col=fea, probabilities=[0.5], relativeError=0.025)[0]} for fea in
        quantitative_features
    }
    print("Current imputing dictionary: ")
    print(imputer_maps)
    print("Analyzing modes")
    imputer_maps.update({
        fea: {'extra_nulls': ['None'],
              'fill_value': df.groupby(fea).count().orderBy("count", ascending=False).first()[0]} for fea in
        ordinal_features + nominal_features
    })
    print("Filling dictionary: ")
    print(imputer_maps)
    # Convert to JSON and save it
    json_data = json.dumps(imputer_maps, indent=4)

    # Save to a file
    with open(os.path.join(dir_save_params, 'imputer_maps.json'), 'w') as f:
        f.write(json_data)

    # ----------------------------- NOMINAL ENCODER ----------------------------

    def get_sufficiently_frequent(df, fea, frequency_threshold=frequency_threshold):
        total_count = df.count()

        # Group by the column and calculate the normalized frequency
        proportions = df.groupBy(fea).agg(
            (F.count("*") / total_count).alias(f"{fea}_frequency")
        )
        result = proportions.filter(F.col(f"{fea}_frequency") > frequency_threshold).select(fea).collect()
        result = [row[fea] for row in result]
        return result

    feature_to_sufficiently_frequent = {
        fea: get_sufficiently_frequent(df, fea) for fea in nominal_features
    }
    print("Sufficiently frequent values per feature: ")
    print(feature_to_sufficiently_frequent)

    # Map between feature and the encoder and new column name
    nominal_encode_type = {}
    nominal_encoders = {}
    new_nominal = []
    for fea in nominal_features:
        elems_to_preserve = feature_to_sufficiently_frequent[fea]
        df = df.withColumn(
            f"{fea}_aggregated",
            (F.when(~F.col(fea).isin(elems_to_preserve), lit("Other")).otherwise(F.col(fea)))
        )

        if len(elems_to_preserve) + 1 <= cardinality_threshold:
            print(f"Performing One-Hot-Encoding to feature {fea}")
            indexer = StringIndexer(inputCol=f"{fea}_aggregated", outputCol=f"{fea}_index", handleInvalid='keep')
            encoder = OneHotEncoder(inputCol=f"{fea}_index", outputCol=f"{fea}_binary", handleInvalid='keep',
                                    dropLast=True)
            pipeline = Pipeline(stages=[indexer, encoder])
            pipeline_model = pipeline.fit(df)
            nominal_encode_type[f"{fea}_aggregated"] = "binary"
            new_nominal.append(f"{fea}_binary")
            pipeline_model.save(os.path.join(dir_save_params, f'{fea}_aggregated_encoder'))
        elif high_cardinality_strategy == "ignore":
            print(f"Ignoring feature {fea}")
        elif high_cardinality_strategy == "mean":
            print(f"Performing Mean-Target-Encoding to feature {fea}")
            mapping_df = df.groupBy(f"{fea}_aggregated").agg(F.avg("ArrDelay").alias(f"{fea}_mean_enc"))
            if "Other" not in mapping_df.select(f"{fea}_aggregated").distinct().collect():
                mean = float(df.groupBy(TARGET_COLUMN).agg(F.avg("ArrDelay")).collect()[0][0])
                print(mean)
                new_row = Row(f"{fea}_aggregated", f"{fea}_mean_enc")("Other", mean)
                print(new_row)
                # Convert the new row to a DataFrame with the same schema as mapping_df
                new_row_df = spark.createDataFrame([new_row], mapping_df.schema)
                print(new_row_df.show())
                mapping_df = mapping_df.union(new_row_df)
                print(mapping_df.show())
            mapping_df.write.csv(os.path.join(dir_save_params, f'{fea}_aggregated_encoder.csv'), header=True)
            new_nominal.append(f"{fea}_mean_enc")
            nominal_encode_type[f"{fea}_aggregated"] = "mean"
        else:
            raise NotImplementedError(f"Not implemented strategy {high_cardinality_strategy}")

    print("Feature to encoder types:")
    print(nominal_encode_type)
    print("Final nominal variables:")
    print(new_nominal)

    # Convert to JSON and save it
    json_data = json.dumps(nominal_encode_type, indent=4)
    with open(os.path.join(dir_save_params, 'encode_types.json'), 'w') as f:
        f.write(json_data)

    json_data = json.dumps(feature_to_sufficiently_frequent, indent=4)
    with open(os.path.join(dir_save_params, 'non_aggregated.json'), 'w') as f:
        f.write(json_data)

    # -------------------------------- VECTORIZER --------------------------------
    # Quantitative feature assembly
    quant_assembler = VectorAssembler(
        inputCols=quantitative_features,
        outputCol="quant_features_vector"
    )

    # Assemble encoded nominal features
    nominal_assembler = VectorAssembler(
        inputCols=new_nominal,
        outputCol="nominal_features_vector"
    )

    ordinal_assembler = VectorAssembler(
        inputCols=ordinal_features,
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
    vectorizer = pipeline.fit(df)
    vectorizer.save(os.path.join(dir_save_params, 'vectorizer'))
    # -------------------------------- VECTORIZER --------------------------------


def dynamic_preprocess(df, nominal_features, ordinal_features, quantitative_features, dir_load_params):
    # -------------------------------- IMPUTER --------------------------------
    with open(os.path.join(dir_load_params, 'imputer_maps.json'), 'r') as f:
        imputer_maps = json.load(f)

    for fea in quantitative_features + ordinal_features + nominal_features:
        if len(imputer_maps[fea]['extra_nulls']) > 0:
            df = df.withColumn(fea, when(df[fea].isin(imputer_maps[fea]['extra_nulls']), lit(None)).otherwise(df[fea]))

        if df.filter(col(fea).isNull()).count() > 0:
            value = imputer_maps[fea]['fill_value']
            print(f"Imputing {fea} with {value}")
            df = df.fillna(value, subset=fea)

    # ----------------------------- NOMINAL ENCODER ----------------------------
    with open(os.path.join(dir_load_params, 'encode_types.json'), 'r') as f:
        encode_types = json.load(f)
    with open(os.path.join(dir_load_params, 'non_aggregated.json'), 'r') as f:
        fea_2_non_aggregated = json.load(f)

    for fea, non_aggregated in fea_2_non_aggregated.items():
        df = df.withColumn(
            f"{fea}_aggregated",
            (F.when(~F.col(fea).isin(non_aggregated), lit("Other")).otherwise(F.col(fea)))
        )

    for fea, encode_type in encode_types.items():
        if encode_type == 'binary':
            encoder = PipelineModel.load(os.path.join(dir_load_params, f'{fea}_encoder'))
            df = encoder.transform(df)
        elif encode_type == 'mean':
            encoder = SparkSession.builder.getOrCreate().read.csv(os.path.join(dir_load_params, f'{fea}_encoder.csv'),
                                                                  header=True, inferSchema=True)
            df = df.join(encoder, on=fea, how='left')
            new_var = f"{fea}_mean_enc".replace("_aggregated", "")
            imput_value = encoder.filter(encoder[fea] == "Other").select(new_var).collect()[0][0]
            print(f"Using the following encoder for {fea}")
            print(encoder.show(10))
            print(f"Imputing unrecognized values in {fea} with 'Other'->{imput_value}")
            df = df.fillna(imput_value, subset=new_var)
        else:
            raise NotImplementedError(f"Not implemented encode type {encode_type}")

    # ------------------------------ VECTORIZER --------------------------------
    vectorizer = PipelineModel.load(os.path.join(dir_load_params, 'vectorizer'))
    df = vectorizer.transform(df)
    return df


def assure_existence_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def preprocess_fit_and_transform(df, df_planes, dir_save_params="./data/"):
    df, quantitative_features, ordinal_features, nominal_features = static_preprocess(df, df_planes)

    if len(os.listdir(dir_save_params)) == 0:
        print("TRAINING DYNAMIC PREPROCESSING PARAMETERS")
        train_preprocess(df, nominal_features, ordinal_features, quantitative_features, dir_save_params,
                         cardinality_threshold=10, frequency_threshold=0.02, high_cardinality_strategy="mean")
    else:
        print("DYNAMIC PREPROCESSING PARAMETERS FOUND. SKIPPING LEARNING.")
    df = dynamic_preprocess(df, nominal_features, ordinal_features, quantitative_features, dir_save_params)
    return df


def split_and_preprocess(df, df_planes, train_frac=0.8, dir_save_params="./data/"):
    train_df, test_df = df.randomSplit([train_frac, 1 - train_frac], seed=42)
    train_df = preprocess_fit_and_transform(train_df, df_planes, dir_save_params=dir_save_params)

    print("TESTING DATA PROCESSING")
    test_df, quantitative_features, ordinal_features, nominal_features = static_preprocess(test_df, df_planes)
    test_df = dynamic_preprocess(test_df, nominal_features, ordinal_features, quantitative_features, dir_save_params)
    return train_df, test_df


def validation_preprocess(df, df_planes, dir_save_params="./data/"):
    df, quantitative_features, ordinal_features, nominal_features = static_preprocess(df, df_planes)
    df = dynamic_preprocess(df, nominal_features, ordinal_features, quantitative_features, dir_save_params)
    return df


def main(n_partitions=10, debug=False):
    spark = (SparkSession.builder.appName("MachineLearningProject")
             .config("spark.executor.memory", "4g")
             .config("spark.driver.memory", "48g")
             .config("spark.memory.fraction", "0.8")
             .config("spark.memory.storageFraction", "0.3")
             .config("spark.driver.maxResultSize", "4g")
             .config("spark.sql.caseSensitive", "true")
             .config("spark.sql.debug.maxToStringFields", "200")
             # .config("spark.local.dir", "./temp/")
             .getOrCreate())

    if not os.path.exists(PROCESSED_TRAIN_PARQUET):
        df, df_planes = load(spark, FLIGHT_PARQUET_PATH, PLANE_SCHEMA_PATH, FLIGHT_RAW_PATH)
        df = df.repartition(n_partitions)

        if debug:
            fraction = 0.01  # Adjust the fraction to select 10% of rows
            df = df.sample(withReplacement=True, fraction=fraction)
            df = df.repartition(1)

        # train_df, test_df = complete_preprocess(df, df_planes, train_frac=0.8)
        assure_existence_directory(PROCESSING_DIR)
        train_df, test_df = split_and_preprocess(df, df_planes, train_frac=0.8, dir_save_params=PROCESSING_DIR)
        print("Finished preprocessing")
        print(train_df.head())
        print(test_df.head())

        print(f"Saving schema to {PROCESSED_SCHEMA}")
        assure_existence_directory(PROCESSED_DIR)
        schema_json = train_df.schema.json()
        with open(PROCESSED_SCHEMA, 'w') as f:
            f.write(schema_json)
        test_df.write.mode('overwrite').parquet(PROCESSED_TEST_PARQUET)
        train_df.write.mode('overwrite').parquet(PROCESSED_TRAIN_PARQUET)
    else:
        with open(PROCESSED_SCHEMA, 'r') as f:
            schema_json = f.read()

        schema = StructType.fromJson(json.loads(schema_json))

        test_df = spark.read.parquet(PROCESSED_TEST_PARQUET, schema=schema)
        train_df = spark.read.parquet(PROCESSED_TRAIN_PARQUET, schema=schema)

        print(test_df.head())
        print(train_df.head())
    spark.stop()


if __name__ == "__main__":
    import os

    os.environ['PYSPARK_PYTHON'] = r'C:\Users\franb\AppData\Local\Programs\Python\Python38\python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\franb\AppData\Local\Programs\Python\Python38\python.exe'
    # os.environ["PATH"] = r"C:\Users\franb\AppData\Local\Programs\Python\Python38" + os.pathsep + os.environ["PATH"]

    main(debug=False)

"""
class CustomMeanEncoder(Transformer,  # Base class
                        HasInputCol,  # Sets up an inputCol parameter
                        HasOutputCol,  # Sets up an outputCol parameter
                        DefaultParamsReadable,  # Makes parameters readable from file
                        DefaultParamsWritable  # Makes parameters writable from file
                        ):
    # append_str is a value which we would like to be able to store state for, so we create a parameter.
    mapping = Param(
        Params._dummy(),
        "mapping",
        "Mapping from nominal key to quantitative value",
        # typeConverter=TypeConverters.toString,  # This will allow code to automatically try to convert to string
    )
    default_key = Param(
        Params._dummy(),
        "default_key",
        "Default nominal key",
        # typeConverter=TypeConverters.toString,  # This will allow code to automatically try to convert to string
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, mapping=None, default_key=None):
        
        # Constructor: set values for all Param objects
        
        # self._input_kwargs = {}
        super().__init__()
        self._setDefault(mapping=None)
        self._setDefault(default_key=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, mapping=None, default_key=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setMapping(self, mapping):
        return self.setParams(mapping=mapping)

    def setDefault(self, default_key):
        return self.setParams(default_key=default_key)

    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)

    def _transform(self, dataset):
        
        # This is the main member function which applies the transform to transform data from the `inputCol` to the `outputCol`
    
        if not self.isSet("inputCol"):
            raise ValueError(
                "No input column set for the "
                "CustomMeanEncoding transformer."
            )
        input_column = self.getInputCol()
        output_column = self.getOutputCol()
        df = self.getOrDefault(self.mapping)
        default_key = self.getOrDefault(self.default_key)

        # Broadcast the dictionary to improve efficiency
        print("Mapping using ", df.head())

        print("Join and fill...", end="")
        dataset = dataset.join(df, on=input_column, how='left')
        imput_value = df.filter(df["key"] == default_key).select("value").collect()[0]
        dataset = dataset.fillna(imput_value)
        print("done.")
        return dataset

    def save(self, path):
        # Save the transformer state to a file
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # Load the transformer state from a file
        with open(path, 'rb') as f:
            return pickle.load(f)
            
def aggregate_infrequent(train_df, test_df, fea, frequency_threshold=0.05):
    total_count = train_df.count()

    # Group by the column and calculate the normalized frequency
    proportions = train_df.groupBy(fea).agg(
        (F.count("*") / total_count).alias(f"{fea}_frequency")
    )
    not_merged = proportions.filter(F.col(f"{fea}_frequency") > frequency_threshold).select(fea).collect()
    not_merged = [row[fea] for row in not_merged]
    print(f"Merging in feature {fea} those not in: {not_merged}")
    train_df = train_df.withColumn(
        f"{fea}_merged",
        (F.when(~F.col(fea).isin(not_merged), lit("Infrequent")).otherwise(F.col(fea)))
    )
    test_df = test_df.withColumn(
        f"{fea}_merged",
        (F.when(~F.col(fea).isin(not_merged), lit("Infrequent")).otherwise(F.col(fea)))
    )
    return train_df, test_df, f"{fea}_merged"


def preprocess_nominal(train_df, test_df, nominal_features, cardinality_threshold=10,
                       high_cardinality_strategy="ignore", frequency_threshold=0.05):
    new_nominal = []
    for fea in nominal_features:
        train_df, test_df, fea = aggregate_infrequent(train_df, test_df, fea, frequency_threshold)

        cardinality = train_df.select(fea).distinct().count()
        if cardinality <= cardinality_threshold:
            print(f"Performing One-Hot-Encoding to feature {fea}")
            indexer = StringIndexer(inputCol=f"{fea}", outputCol=f"{fea}_index", handleInvalid='keep')
            encoder = OneHotEncoder(inputCol=f"{fea}_index", outputCol=f"{fea}_binary", handleInvalid='keep',
                                    dropLast=True)
            pipeline = Pipeline(stages=[indexer, encoder])
            pipeline_model = pipeline.fit(train_df)
            train_df = pipeline_model.transform(train_df)
            test_df = pipeline_model.transform(test_df)
            train_df = train_df.drop(f"{fea}_index")
            test_df = test_df.drop(f"{fea}_index")
            new_nominal.append(f"{fea}_binary")
        elif high_cardinality_strategy == "ignore":
            print(f"Ignoring feature {fea}")
        elif high_cardinality_strategy == "mean":
            print(f"Performing Mean-Target-Encoding to feature {fea}")
            fea_means_map = train_df.groupBy(fea).agg(F.avg("ArrDelay").alias(f"{fea}_mean_enc"))
            train_df = custom_mean_encoding(train_df, fea_means_map, fea)
            test_df = custom_mean_encoding(test_df, fea_means_map, fea)
            new_nominal.append(f"{fea}_mean_enc")
        else:
            raise NotImplementedError(f"Not implemented strategy {high_cardinality_strategy}")

    print(f"New nominal variables: {new_nominal}")
    print(test_df.head())
    print(train_df.head())
    return train_df, test_df, new_nominal


def complete_preprocess(flight_df, df_planes, train_frac, cardinality_threshold=15, high_cardinality_strategy="mean",
                        frequency_threshold=0.02):
    df_planes = df_planes.withColumnRenamed("tailnum", "TailNum")
    flight_df = flight_df.join(df_planes, on="TailNum", how="inner")

    flight_df, quantitative_features, ordinal_features, nominal_features = static_preprocess(flight_df)

    train_df, test_df = flight_df.randomSplit([train_frac, 1 - train_frac], seed=42)

    print("Current schema")
    train_df.printSchema()

    for fea in nominal_features + ordinal_features:
        train_df = train_df.withColumn(fea, when(train_df[fea] == 'None', lit(None)).otherwise(train_df[fea]))
        test_df = test_df.withColumn(fea, when(test_df[fea] == 'None', lit(None)).otherwise(test_df[fea]))

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
    # print("Printing null counts")
    # null_counts = train_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in
    #                                quantitative_features + nominal_features + ordinal_features])
    # null_counts.show()

    train_df, test_df, nominal_features = preprocess_nominal(train_df, test_df, nominal_features,
                                                             cardinality_threshold=cardinality_threshold,
                                                             high_cardinality_strategy=high_cardinality_strategy,
                                                             frequency_threshold=frequency_threshold)

    # Quantitative feature assembly
    quant_assembler = VectorAssembler(
        inputCols=quantitative_features,
        outputCol="quant_features_vector"
    )

    # Assemble encoded nominal features
    nominal_assembler = VectorAssembler(
        inputCols=nominal_features,
        outputCol="nominal_features_vector"
    )

    ordinal_assembler = VectorAssembler(
        inputCols=ordinal_features,
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

"""
