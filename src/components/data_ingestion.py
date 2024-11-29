import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data-ingestion', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info(f"Data Ingestion config initialized with path: {self.ingestion_config.raw_data_path}")

    def initiate_data_ingestion(self):
        logging.info("Started data ingestion process")
        try:
            logging.info("Attempting to read dataset from data-source directory")
            df = pd.read_csv('data-source/WA_Fn-UseC_-Telco-Customer-Churn.csv')
            logging.info(f"Dataset read successfully with shape: {df.shape}")
            logging.info(f"Dataset columns: {df.columns.tolist()}")
            
            logging.info(f"Creating directory at: {os.path.dirname(self.ingestion_config.raw_data_path)}")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            logging.info("Directory created successfully")
            
            logging.info("Starting data cleaning process")
            logging.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")
            
            logging.info("Converting TotalCharges column to numeric")
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
            logging.info(f"Missing values in TotalCharges after conversion: {df['TotalCharges'].isnull().sum()}")
            
            logging.info("Removing customerID column")
            df = df.drop('customerID', axis=1)
            logging.info(f"Remaining columns after dropping customerID: {df.columns.tolist()}")
            
            logging.info(f"Dataset statistics after cleaning:")
            logging.info(f"Number of rows: {df.shape[0]}")
            logging.info(f"Number of columns: {df.shape[1]}")
            logging.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")
            logging.info(f"Number of customers churned: {df['Churn'].value_counts()['Yes']}")
            logging.info(f"Churn rate: {(df['Churn'].value_counts()['Yes']/len(df))*100:.2f}%")
            
            logging.info(f"Saving processed data to {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Data saved successfully")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {str(e)}")
            raise CustomException(e, sys)