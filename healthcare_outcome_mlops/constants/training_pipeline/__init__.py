import os
import sys
import numpy as np
import pandas as pd 

"common constants"

TARGET_COLUMN = "Test Results"
PIPELINE_NAME: str = "HealthcarePipeline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "healthcare.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


"data ingestion"

DATA_INGESTION_COLLECTION_NAME: str = "healthcare_data"
DATA_INGESTION_DATABASE_NAME: str = "healthcare_outcome_mlops"
DATA_INGESTION_DIR_NAME: str = "healthcare_data"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DATA_DIR: str = "ingested_data"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2