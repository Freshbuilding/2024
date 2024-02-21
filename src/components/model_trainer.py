"""
This module defines the process of training ML models for regression tasks.
It includes configuration for saving trained models,
initiation of model training with various regression algorithms, and
evaluation of these models to select the best performing one based on R2 score.
The module leverages several regression algorithms from libraries and
provides functionality to automatically select the best model based
on testing data performance.
"""

import os
import sys
from dataclasses import dataclass

# Importing machine learning libraries and models
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Custom modules for logging, exception handling, and utility functions
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model trainer including the file path
    for saving the trained model.
    """
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl")


class ModelTrainer:
    """
    Handles the training of various regression models,
    evaluation of their performance, and saving the best
    performing model.
    """

    def __init__(self):
        """
        Initializes the ModelTrainer class with default configuration settings.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models with the provided training data,
        evaluates them with the test data, and saves the best model
        based on R2 score.

        Parameters:
            train_array (np.ndarray):
                Training data including features and target.
            test_array (np.ndarray):
                Testing data including features and target.

        Returns:
            float: The R2 score of the best model on the test data.
        """
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training data
                train_array[:, -1],   # Target for training data
                test_array[:, :-1],   # Features for test data
                test_array[:, -1]     # Target for test data
            )

            # Dictionary of models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-neighbors": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Parameters for grid search or model tuning
            params = {
                # Example parameters for "Decision Tree" model
                "Decision Tree": {
                    'criterion': [
                        'squared_error',
                        'friedman_mse',
                        'absolute_error',
                        'poisson'
                    ],
                },
                # Further parameters for other models can be defined here...
            }

            # Evaluate models and obtain performance report
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Find the best model based on the score
            best_model_score = max(model_report.values())
            best_model_name = [
                name for name,
                score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            # Check if the best model meets a minimum score threshold
            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found")

            logging.info("Best model found and saved.")

            # Save the best model to file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict with the best model and calculate R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys) from e
