import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
import mlflow
warnings.filterwarnings('ignore')

# Import MLflow tracking functions
from mlflow_tracking import setup_mlflow, log_parameters, log_metrics, log_model, log_artifacts, get_best_model

def load_and_preprocess_data(file_path):
    """Load and preprocess the churn data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Log data characteristics
    data_params = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'churn_rate': df['Churn'].mean(),
        'missing_values': df.isnull().sum().sum()
    }
    log_parameters(data_params)
    
    return df

def prepare_features(df):
    """Prepare features for model training"""
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'Churn' from numerical columns if present
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform the training data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = (
        numerical_cols +
        [f"{col}_{val}" for col, vals in zip(categorical_cols, preprocessor.named_transformers_['cat'].categories_) for val in vals]
    )
    
    # Log preprocessing parameters
    preprocessing_params = {
        'test_size': 0.2,
        'random_state': 42,
        'numerical_features': numerical_cols,
        'categorical_features': categorical_cols,
        'preprocessing_steps': 'StandardScaler + OneHotEncoder'
    }
    log_parameters(preprocessing_params)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple models"""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': 1.0,
                'max_iter': 1000
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': 5,
                'min_samples_split': 5
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'params': {}
        }
    }
    
    results = {}
    
    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        mlflow.end_run()  # End any existing run
        with mlflow.start_run(run_name=name):
            # Log model parameters
            log_parameters(model_info['params'])
            
            # Train model
            model = model_info['model']
            model.set_params(**model_info['params'])
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            log_metrics(metrics)
            
            # Log the model
            log_model(model, name, X_train)
            
            # Create and log confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name}.png')
            log_artifacts([f'confusion_matrix_{name}.png'])
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
                plt.title(f'Top 10 Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name}.png')
                log_artifacts([f'feature_importance_{name}.png'])
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
    
    return results

def main():
    # Set up MLflow tracking with port 5001
    mlflow.set_tracking_uri("http://0.0.0.0:5001")
    experiment_name = "churn"  # Set fixed experiment name
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment name: {experiment_name}")
    
    # Load and preprocess data
    df = load_and_preprocess_data('/Users/mac/Desktop/DEPI_Fproject/app/cleaned_customer_data.csv')
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)
    
    # Train and evaluate all models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Print comparison of all models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame({
        name: result['metrics'] for name, result in results.items()
    }).T
    print(comparison_df.sort_values('roc_auc', ascending=False))
    
    # Get the best model run
    best_run = get_best_model(experiment_name, metric_name="roc_auc")
    print("\nBest model run:")
    print(best_run)

if __name__ == "__main__":
    main() 