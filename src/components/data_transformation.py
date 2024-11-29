import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'data-transformation', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info(f"Data Transformation config initialized with preprocessor path: {self.data_transformation_config.preprocessor_obj_file_path}")

    def get_data_transformer_object(self):
        try:
            logging.info("Initializing data transformation pipelines")
            
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            logging.info("Creating numerical pipeline with imputer and scaler")
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info("Creating categorical pipeline with imputer and one-hot encoder")
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
                ]
            )

            logging.info("Creating column transformer with numerical and categorical pipelines")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating preprocessor object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path):
        try:
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"Dataset shape: {df.shape}")
            logging.info(f"Dataset columns: {df.columns.tolist()}")

            logging.info("Getting preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info("Creating target variable")
            target_column = 'Churn'
            target_df = df[target_column].map({'Yes': 1, 'No': 0})
            logging.info(f"Target variable distribution:\n{target_df.value_counts()}")
            
            columns_to_drop = [target_column]
            if 'customerID' in df.columns:
                columns_to_drop.append('customerID')
            logging.info(f"Columns to drop: {columns_to_drop}")
                
            input_feature_df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
            logging.info(f"Input features shape after dropping columns: {input_feature_df.shape}")

            logging.info(f"Creating directory: {os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path)}")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            logging.info("Applying preprocessing object on input features")
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)
            logging.info(f"Transformed feature array shape: {input_feature_arr.shape}")

            logging.info("Saving preprocessing object")
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessing_obj, f)
            logging.info(f"Saved preprocessing object at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                input_feature_arr,
                target_df,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)