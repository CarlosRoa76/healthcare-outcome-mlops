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

# Industry standard algorithms for clinical tabular/categorical matrices
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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
        
    def train_model(self, x_train, y_train, x_test, y_test):
        """
        Runs hyperparameter tuning across models and returns the best performing tuned asset.
        """
        logging.info("Entered train_model method of ModelTrainer class")
        try:
            # 1. Define base model architectures
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "RandomForest": RandomForestClassifier(random_state=42),
                "GradientBoosting": GradientBoostingClassifier(random_state=42)
            }

            # 2. Define search spaces for tuning (Keep spaces light for initial execution speed)
            params = {
                "LogisticRegression": {
                    "C": [0.1, 1.0, 10.0]
                },
                "RandomForest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                },
                "GradientBoosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            # Import the utility function we fixed
            from healthcare_outcome_mlops.utils.main_utils.utils import evaluate_models

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

            logging.info(f"Tuning complete. Best Model: {best_model_name} | Accuracy: {best_model_score}")
            return best_model_name, best_model_object

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Initiating model trainer artifact composition logic sequence")
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transform_test_file_path

            # Load clean preprocessed numeric matrices from storage 
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Slicing input engineering features from trailing mapped index targets
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Correctly passed all 4 mandatory positional parameters here 
            best_model_name, best_model = self.train_model(
                x_train=x_train, 
                y_train=y_train, 
                x_test=x_test, 
                y_test=y_test
            )

            # Calculate isolated artifacts across training and testing splits for drift tracking
            y_train_pred = best_model.predict(x_train)
            train_metric_artifact = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            test_metric_artifact = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # FIX: Load the saved preprocessor object from your transformation artifacts path
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # FIX: Pack both the model and preprocessor into your tracking container module
            healthcare_model = HealthcareModel(preprocessor=preprocessing_obj, model=best_model)

            logging.info("Persisting healthcare_model pickle files to directory infrastructure pathways.")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=healthcare_model)

            # Construct final production tracking validation artifact mapping signature
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            
            logging.info(f"Model Trainer operation context finalized. Artifact built at: {model_trainer_artifact.trained_model_file_path}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)