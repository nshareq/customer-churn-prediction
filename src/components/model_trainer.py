import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import pickle
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model-trainer', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            logging.info("MLflow tracking URI set successfully")
            mlflow.set_experiment("customer_churn_prediction")
            logging.info("MLflow experiment initialized")
        except Exception as e:
            logging.error(f"Error initializing MLflow: {str(e)}")
            raise CustomException(e, sys)

    def get_model_params(self):
        logging.info("Getting model parameters")
        models = {
            'decision_tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000]
                }
            }
        }
        logging.info("Model parameters configured successfully")
        return models

    def evaluate_model(self, y_true, y_pred):
        logging.info("Evaluating model performance")
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred)
            }
            conf_matrix = confusion_matrix(y_true, y_pred)
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
            logging.info(f"Model evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_training(self, features, target):
        try:
            logging.info("Starting model training process")
            logging.info(f"Features shape: {features.shape}")
            logging.info(f"Target shape: {target.shape}")

            models = self.get_model_params()
            best_model = None
            best_score = 0
            best_params = None
            model_name = None

            logging.info("Starting model training and evaluation")
            for name, config in models.items():
                logging.info(f"\nTraining {name} model")
                with mlflow.start_run(run_name=name):
                    logging.info(f"MLflow run started for {name}")
                    
                    grid_search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=5,
                        scoring='f1',
                        n_jobs=-1,
                        verbose=2
                    )
                    
                    logging.info(f"Starting GridSearchCV for {name}")
                    grid_search.fit(features, target)
                    logging.info(f"Best parameters for {name}: {grid_search.best_params_}")
                    
                    y_pred = grid_search.predict(features)
                    metrics = self.evaluate_model(target, y_pred)
                    
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(grid_search.best_estimator_, name)
                    logging.info(f"MLflow logging completed for {name}")

                    if metrics['f1_score'] > best_score:
                        best_score = metrics['f1_score']
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        model_name = name
                        logging.info(f"New best model found: {name} with f1_score: {best_score}")

            logging.info("\nModel Training Summary:")
            logging.info(f"Best performing model: {model_name}")
            logging.info(f"Best parameters: {best_params}")
            logging.info(f"Best f1 score: {best_score}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            return best_model, best_score

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)   