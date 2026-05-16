import os
import sys
import numpy as np
import pandas as pd 

"Common Constants"

TARGET_COLUMN = "Test Results"
PIPELINE_NAME: str = "HealthcarePipeline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "healthcare.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"Data Ingestion"

DATA_INGESTION_COLLECTION_NAME: str = "healthcare_data"
DATA_INGESTION_DATABASE_NAME: str = "healthcare_outcome_mlops"
DATA_INGESTION_DIR_NAME: str = "healthcare_data"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DATA_DIR: str = "ingested_data"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"Data Validation"

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"