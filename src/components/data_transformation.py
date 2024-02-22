"""
This module defines the classes and configurations necessary for the data
transformation phase in a machine learning pipeline.
It includes functionality for preprocessing data by imputing missing values,
encoding categorical variables, and scaling features
to prepare them for model training and evaluation.
"""

from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom modules for logging and exception handling
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation processes.

    Attributes:
        preprocessor_obj_file_path:
        The file path where the preprocessing object will be saved.
        This object is responsible for performing all preprocessing
        steps required before feeding data into a machine learning model.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Handles the transformation of data for machine learning models.

    This includes preprocessing steps such as imputation,
    encoding of categorical variables, and scaling of features.
    The class provides functionalities to create a preprocessing pipeline
    and to apply this pipeline to training and testing datasets,
    preparing them for model training and evaluation.

    Methods:
        get_data_transformer_object:
            Constructs and returns a preprocessing pipeline.
        initiate_data_transformation:
            Applies the preprocessing pipeline to the training and
            testing datasets and saves the transformed datasets along
            with the preprocessing object.
    """

    def __init__(self):
        """
        Initializes the DataTransformation class
        with default configuration settings.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor object for transforming data.

        Constructs a preprocessing pipeline that includes handling
        both numerical and categorical features.
        Numerical features are imputed with the median and scaled,
        while categorical features are imputed with the most frequent value,
        one-hot encoded, and scaled.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process
        for both training and testing datasets.

        Reads the datasets from given paths,
        applies the preprocessing transformations,
        and saves the transformed data along with
        the preprocessing object for future use.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
