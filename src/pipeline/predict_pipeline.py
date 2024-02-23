import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """Class for managing the prediction pipeline."""

    def __init__(self):
        """Initialize PredictPipeline object."""
        pass  

    def predict(self, features):
        """
        Predict outcomes based on the input features.
        """
        try:
            # Construct paths for model and preprocessor objects
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print('Loading model and preprocessor objects from artifacts')
            # Load model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print('Model and preprocessor successfully loaded')
            # Preprocess the features and predict outcomes
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    """
    Class for encapsulating custom data with attributes for prediction.
    """

    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        """
        Initialize CustomData object with given attributes.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the CustomData attributes to a pandas DataFrame.

        :return: DataFrame representation of the custom data.
        """
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) from e
