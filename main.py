from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys

def main():
    try:
        logging.info("\n\n>>>>> Starting the Machine Learning Pipeline <<<<<\n")

        # Data Ingestion
        logging.info(">>>>> Initiating Data Ingestion Phase <<<<<")
        data_ingestion = DataIngestion()
        data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Data saved at: {data_path}")

        # Data Transformation
        logging.info("\n>>>>> Initiating Data Transformation Phase <<<<<")
        data_transformation = DataTransformation()
        features_arr, target_arr, preprocessor_path = data_transformation.initiate_data_transformation(data_path)
        logging.info(f"Data transformation completed. Features shape: {features_arr.shape}, Target shape: {target_arr.shape}")

        # Model Training
        logging.info("\n>>>>> Initiating Model Training Phase <<<<<")
        model_trainer = ModelTrainer()
        best_model, best_score = model_trainer.initiate_model_training(features_arr, target_arr)
        logging.info(f"Model training completed. Best model score: {best_score:.4f}")

        logging.info("\n>>>>> Machine Learning Pipeline Completed Successfully <<<<<\n")
        
        return best_model, best_score

    except Exception as e:
        logging.error(">>>>> Error occurred in the Machine Learning Pipeline <<<<<")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        best_model, best_score = main()
        print(f"\nBest Model F1 Score: {best_score:.4f}")
    except Exception as e:
        print(f"\nError: {e}")