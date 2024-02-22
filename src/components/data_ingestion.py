"""
This module is designed to handle data ingestion for a ML project.
It includes functionality to read data from a CSV file,
split it into training and testing datasets, and
save those datasets to specified paths.
"""

import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
# CustomException for handling exceptions specific to this application.
from src.exception import CustomException
# logging for logging messages throughout the data ingestion process.
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion paths.
    Defines paths for training, testing, and raw data.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Handles the ingestion of data from a source, splitting into training and
    test datasets, and saving these datasets to specified paths.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the source dataset, splits it into training and test sets,
        and saves these sets to their respective paths.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a pandas DataFrame.
            # Change this for other sources (MongoDB or other)
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Ensure the directory for saving files exists.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),
                        exist_ok=True)

            # Save the raw data.
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("Train test split initiated")
            # Split the dataset into training and testing sets.
            train_set, test_set = train_test_split(df, test_size=0.2,
                                                   random_state=42)

            # Save the training and testing sets.
            train_set.to_csv(self.ingestion_config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Custom exception for clarity.
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))