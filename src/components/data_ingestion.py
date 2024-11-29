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

    def initiate_data_ingestion(self):
        logging.info("Started data ingestion")
        try:
            # Read dataset
            df = pd.read_csv('data-source/WA_Fn-UseC_-Telco-Customer-Churn.csv')
            logging.info("Dataset read successfully with shape: {df.shape}")

            # Create directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Basic data cleaning
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
            
            # Remove customerID as it's not needed for analysis
            df = df.drop('customerID', axis=1)
            
            # Save data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Data saved to {self.ingestion_config.raw_data_path}")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()