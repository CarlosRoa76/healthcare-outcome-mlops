import os
import sys
import numpy as np
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.logging.logger import logging
from healthcare_outcome_mlops.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from healthcare_outcome_mlops.entity.config_entity import ModelTrainerConfig
from healthcare_outcome_mlops.utils.ml_utils.model.estimator import HealthcareModel

# Imported load_object to read our saved preprocessor transformer instance
from healthcare_outcome_mlops.utils.main_utils.utils import save_object, load_object
from healthcare_outcome_mlops.utils.main_utils.utils import load_numpy_array_data

# Shifted utility from regression to classification metrics
from healthcare_outcome_mlops.utils.ml_utils.metric.classification_metric import get_classification_score
from healthcare_outcome_mlops.utils.main_utils.utils import evaluate_models


# Industry standard algorithms for clinical tabular/categorical matrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

import dagshub
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)


class ModelTrainer:
    def __init__(
            self, 
            model_trainer_config: ModelTrainerConfig,
            data_transformation_artifact: DataTransformationArtifact,
            ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        # Dynamically point to your own repository's MLflow tracking workspace instead of a hardcoded route
        mlflow.set_registry_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run(nested=True):
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision
            recall_score = classificationmetric.recall
            accuracy = classificationmetric.accuracy

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model artifact (Bypasses 404 registration blocks, keeping your logs clean and functional)
            mlflow.sklearn.log_model(best_model, "model")

        
    def train_model(self, x_train, y_train, x_test, y_test):
        """
        Runs hyperparameter tuning across models and returns the best performing tuned asset.
        """
        logging.info("Entered train_model method of ModelTrainer class")
        try:
            # 1. Define base model architectures
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "AdaBoost": AdaBoostClassifier(),
                }

            # 2. Define search spaces for tuning (Keep spaces light for initial execution speed)
            params={
            "Decision Tree": {},
            "Random Forest":{},
            "Gradient Boosting":{},
            "Logistic Regression":{},
            "AdaBoost":{}
            }

            # 3. Run the hyperparameter grid tuning tournament
            model_report: dict = evaluate_models(
                X_train=x_train, y_train=y_train, 
                X_test=x_test, y_test=y_test, 
                models=models, param=params
            )
            
            # 4. Find the highest accuracy score from our report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Extract the actual fitted object of the winning candidate model
            best_model_object = models[best_model_name]
            y_train_pred = best_model_object.predict(x_train)
            
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            self.track_mlflow(best_model_object, classification_train_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Healthcare_model = HealthcareModel(preprocessor=preprocessor, model=best_model_object)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Healthcare_model)

            save_object("final_model/model.pkl", best_model_object)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_train_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Initiating model trainer artifact composition logic sequence")
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transform_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Fixed: train_model returns a ModelTrainerArtifact object, not a tuple (name, model)
            trainer_artifact = self.train_model(
                x_train=x_train, 
                y_train=y_train, 
                x_test=x_test, 
                y_test=y_test
            )
            
            # Extract the actual trained model file out from your local saved files or configs
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            saved_healthcare_model = load_object(file_path=self.model_trainer_config.trained_model_file_path)
            best_model = saved_healthcare_model.model

            y_train_pred = best_model.predict(x_train)
            train_metric_artifact = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            self.track_mlflow(best_model, train_metric_artifact)

            y_test_pred = best_model.predict(x_test)
            test_metric_artifact = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.track_mlflow(best_model, test_metric_artifact)

            logging.info("Persisting healthcare_model pickle files to directory infrastructure pathways.")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            
            logging.info(f"Model Trainer operation context finalized. Artifact built at: {model_trainer_artifact.trained_model_file_path}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)