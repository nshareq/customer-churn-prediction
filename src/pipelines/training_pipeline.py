import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        logging.info("Initializing Training Pipeline")

    def start_training_pipeline(self):
        try:
            logging.info("\n\n====================Training Pipeline Started====================\n")
            
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion()
            data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Data saved at: {data_path}")

            logging.info("\nStarting Data Transformation")
            data_transformation = DataTransformation()
            features_arr, target_arr, preprocessor_path = data_transformation.initiate_data_transformation(data_path)
            logging.info(f"Data Transformation completed")
            logging.info(f"Features shape: {features_arr.shape}")
            logging.info(f"Target shape: {target_arr.shape}")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            logging.info("\nStarting Model Training")
            model_trainer = ModelTrainer()
            model, score = model_trainer.initiate_model_training(features_arr, target_arr)
            logging.info(f"Model training completed with score: {score:.4f}")

            logging.info("\n====================Training Pipeline Completed====================\n")
            return model, score

        except Exception as e:
            logging.error("Error in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        model, score = training_pipeline.start_training_pipeline()
        print(f"Training completed successfully!")
        print(f"Best model score: {score:.4f}")
    except Exception as e:
        print(f"Error: {e}")