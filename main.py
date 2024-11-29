# Example usage in any other file
from src.logger import logging
from src.exception import CustomException
import sys

try:
    # Your code here
    logging.info("Starting the process")
    result = 1/0  # This will raise an exception
except Exception as e:
    logging.error("Division by zero")
    raise CustomException(e, sys)