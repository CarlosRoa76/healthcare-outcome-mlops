import yaml
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.logging.logger import logging
import sys, os
import numpy as np
import pickle

# Swapped from r2_score to classification scoring
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def write_yaml(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_object(file_path: str) -> object:
    try:
        # FIXED TYPO: Changed .exist to .exists
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Tunes hyperparameters using GridSearchCV and returns a report of test scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Running GridSearchCV hyperparameter tuning for: {model_name}")
            
            # cv=3 splits data into 3 folds for cross-validation scoring
            # n_jobs=-1 uses all available processor cores to speed it up
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Assign the best parameters found by grid search to the model
            model.set_params(**gs.best_params_)
            logging.info(f"Best parameters found for {model_name}: {gs.best_params_}")
            
            # Retrain the model with its ideal configurations
            model.fit(X_train, y_train)

            # Evaluate predictions on the validation test split
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)