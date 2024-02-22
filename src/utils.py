import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves an object to a specified file path using pickle.
    :param file_path: The path where the object should be saved.
    :param obj: The object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(str(e), sys.exc_info())


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV and returns their performance.
    :param X_train, y_train: Training data and labels.
    :param X_test, y_test: Testing data and labels.
    :param models: A dictionary of models to evaluate.
    :param param: A dictionary of parameters for GridSearchCV.
    :return: A report of model performance.
    """
    try:
        report = {}
        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(str(e), sys.exc_info())


def load_object(file_path):
    """
    Loads an object from a specified file path using pickle.
    :param file_path: The path from which the object should be loaded.
    :return: The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(str(e), sys.exc_info())
