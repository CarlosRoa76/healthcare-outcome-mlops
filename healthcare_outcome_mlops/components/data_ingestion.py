from healthcare_outcome_mlops.exception.exception import CustomException
from healthcare_outcome_mlops.logging.logger import logging
from healthcare_outcome_mlops.entity.config_entity import DataIngestionConfig
from healthcare_outcome_mlops.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

import os
from dotenv import load_dotenv

# This ensures it finds the .env file in your current working directory
load_dotenv(os.path.join(os.getcwd(), '.env'))

MONGODB_URI = os.getenv("MONGODB_URI")
print(f"DEBUG: Connection string is: {MONGODB_URI}")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGODB_URI)
            collection = self.mongo_client[database_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
                
            df.replace({np.nan:None}, inplace=True)
            return df
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def split_data_into_train_and_test(self,dataframe:pd.DataFrame):
        try:
            train_df, test_df = train_test_split(
                dataframe, test_size = self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Train and test split done successfully")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Created directory: {dir_path}")

            train_df.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

            logging.info("Train and test split saved successfully")
            

        except Exception as e:
            raise CustomException(e,sys)


    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            
            return dataframe
            
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_data_into_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_into_train_and_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )
            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e,sys)