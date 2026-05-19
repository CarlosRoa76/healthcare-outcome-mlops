import sys
from healthcare_outcome_mlops.logging.logger import logging
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.entity.config_entity import DataIngestionConfig,TrainigPipelineConfig, DataValidationConfig, DataTransformationConfig
from healthcare_outcome_mlops.components.data_validation import DataValidation
from healthcare_outcome_mlops.components.data_ingestion import DataIngestion
from healthcare_outcome_mlops.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainigPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Data ingestion completed successfully")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Initiation Completed")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info("Data Transformation Started")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation Completed")
    except Exception as e:
        raise CustomException(e,sys)