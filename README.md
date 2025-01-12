# Flight Delay Prediction Pipeline

## Overview

This repository contains the implementation of a pipeline for predicting flight arrival delays using Apache Spark. The project processes large datasets of flight and aircraft data to train and evaluate machine learning models. The pipeline includes preprocessing, model training, and validation steps, leveraging Spark's distributed computing capabilities for scalability.

---

## Key Features

- **Data Processing**: Efficient preprocessing of raw CSV data into a format suitable for modeling.
- **Machine Learning**: Integration of Spark MLlib models for regression tasks.
- **Scalability**: Handles large datasets using Spark's distributed processing.
- **Dynamic Configuration**: Command-line arguments enable flexible input and output paths.
- **Metrics Calculation**: Evaluation of models using RMSE, MAE, and R².

---

## Requirements

- Python 3.8+
- Apache Spark 3.0+
- PySpark
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare the Environment

Ensure Spark is installed and configured correctly on your system.

### 2. Run the Pipeline

Use `spark-submit` to run the pipeline. The script accepts command-line arguments for data paths and output directories:

```bash
spark-submit --master local[4] flight_delay_pipeline.py \
    --raw_plane ./data/plane-data.csv \
    --raw_flights ./data/2008.csv.bz2 \
    --out ./results/predictions.csv
```

### 3. Arguments

| Argument       | Description                       | Default                    |
|----------------|-----------------------------------|----------------------------|
| `--raw_plane`  | Path to the plane data CSV        | `./data/plane-data.csv`    |
| `--raw_flights`| Path to the flight data CSV       | `./data/2008.csv.bz2`      |
| `--out`        | Output directory for predictions | `./results/predictions.csv` |

## Contributors

- **Sheyla Leyva Sánchez**
- **Francisco de Borja Lozano**
- **Mariajose Franco**

---

## License

See the [LICENSE](LICENSE) file for details.
