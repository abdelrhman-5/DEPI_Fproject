# Customer Churn Prediction

## Project Overview

A machine learning project to predict customer churn using demographic and behavioral data. The project includes data analysis, model development, and a Streamlit web application for interactive predictions.

## Models

Three classification models were implemented:

1. **Logistic Regression**
2. **Decision Tree** (Best performer with 98.8% accuracy)
3. **Naive Bayes**


## Key Features

### Streamlit Web Application

The project includes a Streamlit app with:

- **Prediction Interface**: Input customer data and get churn predictions
- **Data Insights Dashboard**: Explore patterns in customer data
- **Visualization Tools**: Histograms, pie charts, and correlation analysis


### Model Analysis

- Feature importance analysis identifies key churn factors
- Confusion matrices and performance metrics for each model
- Comparison of model performance


## Installation

```shellscript
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```shellscript
streamlit run app.py
```

The application will be available at [http://localhost:8501](http://localhost:8501)

## Project Structure

```plaintext
customer-churn-prediction/
├── app.py                    # Streamlit application
├── dt_model.pkl              # Trained Decision Tree model
├── cleaned_customer_data.csv # Cleaned dataset for insights
└── README.md                 # Project documentation
```

## Future Work

- Model monitoring and retraining
- Additional feature engineering
- Deployment to production environment