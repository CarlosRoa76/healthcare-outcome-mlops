import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from healthcare_outcome_mlops.constants.training_pipeline import TARGET_COLUMN
from healthcare_outcome_mlops.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from healthcare_outcome_mlops.constants import training_pipeline
from healthcare_outcome_mlops.entity.config_entity import DataTransformationConfig
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.logging.logger import logging
from healthcare_outcome_mlops.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_transformer_object(self) -> ColumnTransformer:
        
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numerical_cols = ['Age', 'Billing Amount', 'Room Number']
            categorical_cols = [
                'Gender', 'Blood Type', 'Medical Condition', 
                'Insurance Provider', 'Admission Type', 'Medication'
            ]
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=training_pipeline.DATA_TRANSFORMATION_NUMERICAL_IMPUTE_STRATEGY)),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            logging.info("Successfully constructed the feature preprocessing pipeline object.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Beginning data extraction and preparation steps.")

            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Separate target and training inputs
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # FIX: Map text target classes to explicit numbers
            target_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
            target_feature_train_df = target_feature_train_df.map(target_mapping)

            # Separate target and testing inputs
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.map(target_mapping)

            preprocessor = self.get_data_transformer_object()

            logging.info("Executing Fit-Transform matrix alterations on datasets.")
            # Train and fit processor on the Training subset
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)
             
            # Horizontal stacking now pairs pure numeric arrays flawlessly
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            logging.info("Creating directory pathways and persisting file artifacts.")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            os.makedirs("final_model", exist_ok=True)

            # Persist processed matrix entities using main_utils helper configurations
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object("final_model/preprocessor.pkl", preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transform_test_file_path=self.data_transformation_config.transformed_test_file_path 
            )
            
            logging.info("Exiting data transformation layer successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)