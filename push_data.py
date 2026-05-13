import os 
import sys
import json
import pandas as pd
import numpy as np
import pymongo
import logging

## from src.healthcare_outcome_mlops.exception.exception import CustomException
## from src.healthcare_outcome_mlops.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where() #secure connection to mongodb // certificate authority

class HealthcareDataExtract():
    def __init__(self):
        pass

    def cv_to_json(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(inplace=True, drop=True)
            
            records = data.to_dict(orient="records")
            return records
            
        except Exception as e:
            logging.error(f"Error in csv_to_json method: {e}")
            raise e
            # raise CustomException(e, sys)

    def insert_data_to_mongodb(self, records, database_name, collection_name):
        try:
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)

            database = client[database_name]
            collection = database[collection_name]

            collection.insert_many(records)

            logging.info(f"Data inserted successfully into MongoDB collection: {collection_name}")
            return(len(records))
        except Exception as e:
            logging.error(f"Error in insert_data_to_mongodb method: {e}")
            raise e
            #raise CustomException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "healthcare_data/healthcare_dataset.csv"
    DATABASE = "healthcare_outcome_mlops"
    Collection = "healthcare_data"

    healthcareobj = HealthcareDataExtract()

    records_list = healthcareobj.cv_to_json(file_path=FILE_PATH)

    no_of_records = healthcareobj.insert_data_to_mongodb(
        records=records_list,
        database_name=DATABASE,
        collection_name=Collection
    )
    logging.info(f"Number of records inserted into MongoDB: {no_of_records}")
    print(f"Number of records inserted into MongoDB: {no_of_records}")