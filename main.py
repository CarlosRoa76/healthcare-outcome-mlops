from healthcare_outcome_mlops.components.data_ingestion import DataIngestion
from healthcare_outcome_mlops.logging.logger import logging
from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.entity.config_entity import DataIngestionConfig,TrainigPipelineConfig
import sys


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainigPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Data ingestion completed successfully")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print("data_ingestion_artifact",data_ingestion_artifact)
    except Exception as e:
        raise CustomException(e,sys)