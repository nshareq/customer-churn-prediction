import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pickle

class PredictionPipeline:
    def __init__(self):
        logging.info("Initializing Prediction Pipeline")
        self.model_path = os.path.join('artifacts', 'model-trainer', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'data-transformation', 'preprocessor.pkl')

    def load_objects(self):
        try:
            logging.info("Loading preprocessor and model")
            
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                logging.info("Preprocessor loaded successfully")

            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
                logging.info("Model loaded successfully")

            return preprocessor, model

        except Exception as e:
            logging.error("Error in loading objects")
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            logging.info("Starting prediction process")
            preprocessor, model = self.load_objects()
            
            logging.info(f"Input features shape: {features.shape}")
            transformed_features = preprocessor.transform(features)
            logging.info(f"Transformed features shape: {transformed_features.shape}")
            
            predictions = model.predict(transformed_features)
            probabilities = model.predict_proba(transformed_features)
            
            logging.info(f"Predictions completed: {len(predictions)} predictions made")
            return predictions, probabilities

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 gender: str,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 tenure: int,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float):
        
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'SeniorCitizen': [self.SeniorCitizen],
                'Partner': [self.Partner],
                'Dependents': [self.Dependents],
                'tenure': [self.tenure],
                'PhoneService': [self.PhoneService],
                'MultipleLines': [self.MultipleLines],
                'InternetService': [self.InternetService],
                'OnlineSecurity': [self.OnlineSecurity],
                'OnlineBackup': [self.OnlineBackup],
                'DeviceProtection': [self.DeviceProtection],
                'TechSupport': [self.TechSupport],
                'StreamingTV': [self.StreamingTV],
                'StreamingMovies': [self.StreamingMovies],
                'Contract': [self.Contract],
                'PaperlessBilling': [self.PaperlessBilling],
                'PaymentMethod': [self.PaymentMethod],
                'MonthlyCharges': [self.MonthlyCharges],
                'TotalCharges': [self.TotalCharges]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created successfully")
            return df

        except Exception as e:
            logging.error("Error in creating DataFrame")
            raise CustomException(e, sys)