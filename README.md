# DEPI_Fproject

## Project Overview
This project predicts customer churn using various machine learning models and tracks experiments with MLflow. The workflow includes data preprocessing, model training, evaluation, and experiment tracking.

## How to Run
1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the MLflow Tracking Server** (if not already running):
   ```bash
   mlflow server --host 0.0.0.0 --port 5001
   ```

3. **Run the churn MLflow script:**
   ```bash
   python3 app/churn_mlflow.py
   ```

## Viewing Results
- Open your browser and go to: [http://0.0.0.0:5001](http://0.0.0.0:5001)
- Select the `churn` experiment to view all runs, metrics, artifacts, and registered models.

## Main Script
- `app/churn_mlflow.py`: Loads and preprocesses data, trains multiple models, logs results and artifacts to MLflow, and compares model performance.

## Data
- Place your cleaned data CSV at: `app/cleaned_customer_data.csv`

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## Example Output
After running the script, you will see model comparison metrics in the terminal and detailed experiment tracking in the MLflow UI.