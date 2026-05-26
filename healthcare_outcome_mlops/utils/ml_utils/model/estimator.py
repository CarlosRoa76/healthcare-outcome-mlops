import os, sys
from healthcare_outcome_mlops.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.logging.logger import logging

class HealthcareModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, x):
        try: 
            x_transform = self.preprocessor.traansform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomException(e, sys)
