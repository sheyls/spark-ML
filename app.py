import argparse
import traceback
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from standalone_data_preprocessing import *


def validate(data, model_path, output_path):
    """
    Loads trained models from disk, generates predictions on new data, evaluates model performance,
    and saves the predictions to CSV files.

    Parameters:
    -----------
    data (pyspark.sql.DataFrame):
        Input Spark DataFrame containing:
        - All feature columns used during training
        - 'ArrDelay' column if evaluation metrics are needed
        DataFrame should be preprocessed using the same steps as training data

    model_path (str):
        Directory path containing the saved models
        Each subdirectory should contain a PipelineModel saved by build_and_train_model()

    output_path (str):
        Directory path where prediction CSV files will be saved
        One CSV file will be created for each model with format: {model_name}_pred.csv

    Evaluation Metrics:
    ------------------
    For each model, calculates:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - R² (R-squared score)

    Output Files:
    ------------
    Creates CSV files containing:
    - Original columns (excluding engineered features)
    - Prediction column with model's delay predictions
    - Excludes intermediate columns (*_features, scaledFeatures)

    Returns:
        None
    """
    # Load the trained model
    name = " ".join(model_path.split('_')[1:])
    model_folder = model_path
    print(f"Loading model: {model_folder}")
    model = PipelineModel.load(model_folder)

    # Make predictions on the input data
    print("Validating model...")
    predictions = model.transform(data)
    rmse_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
    metrics = {}
    metrics['rmse'] = rmse_evaluator.evaluate(predictions)
    print(f"{name} - Root Mean Square Error (RMSE) on test data: {metrics['rmse']}")

    # Mean Absolute Error (MAE)
    mae_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")
    metrics['mae'] = mae_evaluator.evaluate(predictions)
    print(f"{name} - Mean Absolute Error (MAE) on test data: {metrics['mae']}")

    # R-Squared (R²)
    r2_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")
    metrics['r2'] = r2_evaluator.evaluate(predictions)
    print(f"{name} - R-Squared (R²) on test data: {metrics['r2']}")
    # Save predictions to the specified output path

    print(f"Saving predictions to {output_path}")
    old_columns = [col for col in predictions.columns if "_" not in col]
    predictions = predictions.select(old_columns).drop(*["features", "scaledFeatures"])
    predictions.write.mode("overwrite").csv(output_path, header=True)
    print(f"Predictions saved to: {output_path}")


def main(debug=True):
    """
    Main function to execute the pipeline workflow.
    Accepts command-line arguments for dynamic input/output handling.
    """
    try:
        print("Running runnable...")

        parser = argparse.ArgumentParser(description="Flight Delay Prediction Application")
        parser.add_argument("--raw_plane", type=str, required=False,
                            help="Path to load the raw CSV data on plane data", default="./data/plane-data.csv")
        parser.add_argument("--out", type=str, required=False,
                            help="Path to the produced prediction CSV files", default="./out.csv")
        parser.add_argument("--raw_flights", type=str, required=False, default="./data/2008.csv.bz2",
                            help="Path to load the raw CSV data on trips used to predict delay")

        args = parser.parse_args()
        print(args)

        # Start Spark Session
        if not debug:
            spark = SparkSession.builder.appName("FlightDelayPipeline").getOrCreate()
            spark.sparkContext.setLogLevel("WARN")
        else:
            spark = (SparkSession.builder.appName("FlightDelayPipeline")
                     .config("spark.executor.memory", "4g")
                     .config("spark.driver.memory", "48g")
                     .config("spark.memory.fraction", "0.8")
                     .config("spark.memory.storageFraction", "0.3")
                     .config("spark.driver.maxResultSize", "4g")
                     .config("spark.sql.caseSensitive", "true")
                     .config("spark.sql.debug.maxToStringFields", "200").getOrCreate())

        # Paths provided by args
        plane_raw_path = args.raw_plane
        flight_raw_path = args.raw_flights
        aux_dir = "./temp/"

        # Ensure the aux directory exists
        os.makedirs(aux_dir, exist_ok=True)

        # Define paths within aux_dir
        plane_parquet_path = os.path.join(aux_dir, "planes.parquet")
        plane_schema_path = os.path.join(aux_dir, "plane_schema.json")
        flight_parquet_path = os.path.join(aux_dir, "flights.parquet")
        flight_schema_path = os.path.join(aux_dir, "flight_schema.json")

        print(f"Reading files {plane_raw_path} and {flight_raw_path}")
        df_planes = load_csv_save_parquet(spark, plane_raw_path, plane_parquet_path, plane_schema_path)
        df_planes = df_planes.repartition(10)
        df = load_csv_save_parquet(spark, flight_raw_path, flight_parquet_path, flight_schema_path)

        preprocessing_dir = "./best_model/processing/"
        print("Preprocessing data...")
        val_df = validation_preprocess(df, df_planes, preprocessing_dir)

        model_path = "./best_model/retrained_Linear_Regression"
        print("Validating model with data...")
        validate(val_df, model_path, args.out)

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
