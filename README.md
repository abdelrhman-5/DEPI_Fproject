# Customer Churn Prediction Project

## Overview

This project aims to predict customer churn using machine learning techniques based on customer behavioral and demographic data. The project includes data preprocessing, model training, evaluation, and an interactive Streamlit app for predictions and insights.

### Key Features
- **Models Used**: Decision Tree, Logistic Regression, Naive Bayes, Stacking Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Churn distribution, feature distributions, correlation heatmaps, prediction probabilities
- **Deployment**: Streamlit app for real-time predictions and churn analysis

## Dataset

The dataset (`customer_churn_dataset-training-master.csv`) consists of **440,833 rows** and **12 columns**:
- `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, `Churn`

### Preprocessing Steps
1. **Missing Values**: Dropped 1 row with missing data.
2. **Feature Engineering**: Added `Tenure_Category` (e.g., "New (≤6 months)", "Long-term (>3 years)") based on `Tenure`.
3. **Encoding**:
   - One-hot encoding: `Gender`, `Contract Length`, `Tenure_Category`
   - Label encoding: `Subscription Type`
4. **Scaling**: Applied `StandardScaler` to numerical features.

## Model Selection and Evaluation

Multiple models were trained and evaluated. The **Stacking Classifier** (combining Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression) outperformed others.

### Performance Metrics
- **Stacking Classifier**:
  - Accuracy: **98.5%**
  - Precision: **99.0%**
  - Recall: **98.0%**
  - F1-Score: 98.5%
  - ROC-AUC: 99.0%
- **Decision Tree (Tuned)**:
  - Accuracy: **98.8%**
  - Precision: **98.0%**
  - Recall: **97.5%**
- **Logistic Regression**: Accuracy: **89.22%**
- **Naive Bayes**: Accuracy: **91.5%**

## Churn Analysis Using Visuals

The project includes several visualizations for churn analysis, implemented in the notebook and Streamlit app:

1. **Churn Distribution**:
   - Pie chart showing **56.7% churned** vs. **43.3% stayed**.
2. **Feature Distributions**:
   - Histograms for `Total Spend`, `Support Calls`, `Age`, `Payment Delay`.
3. **Correlation Heatmap**:
   - Heatmap of numerical features, highlighting correlations (e.g., `Support Calls` and `Payment Delay` positively correlated with churn).
4. **Prediction Probability**:
   - Bar chart displaying churn probability for individual predictions in the Streamlit app.
5. **Learning Curve**:
   - Plot in the notebook showing training and validation accuracy as training set size increases.

## Deployment

The Streamlit app (`deployment.py`) provides an interactive interface with three sections:
- **Predict Churn**: Enter customer details (e.g., `Age`, `Support Calls`) to predict churn probability using the Decision Tree model (loaded from `dt_model.pkl`).
- **Insights**: Visualize churn distribution, feature distributions, gender distribution, and correlation heatmaps; filter data by churn status and download results.
- **About**: Details on the model and its ~98% accuracy.

### Running the Streamlit App
1. Install dependencies:
   ```bash
   pip install streamlit pandas joblib matplotlib seaborn
   ```
2. Run the app:
   ```bash
   streamlit run deployment.py
   ```

## Usage Instructions

- **Prediction**: Input customer data in the "Predict Churn" section and click "Predict" to see churn likelihood and probability.
- **Insights**: Explore visualizations and filter data by churn status ("All", "Stay (0)", "Churn (1)").
- **Download**: Export filtered data as a CSV file from the "Insights" section.

## Key Insights

- **Churn Rate**: 56.7% of customers churned.
- **Top Predictors**: `Support Calls`, `Payment Delay`, `Total Spend`.
- **Correlations**: Positive correlation between churn and `Support Calls` (0.57) and `Payment Delay` (0.31); negative correlation with `Total Spend` (-0.43).

## Notebook Highlights

The Jupyter notebook (`churn_pre.ipynb`) uses markdown for structure and includes:
- **Data Exploration**: Summary stats, missing value checks, class distribution.
- **Preprocessing**: Code blocks for encoding, scaling, and feature engineering.
- **Modeling**: Pipeline setup, hyperparameter tuning with `GridSearchCV`, and model evaluation.
- **Visualizations**: Learning curve plot and correlation analysis.

## License

This project is licensed under the MIT License.
