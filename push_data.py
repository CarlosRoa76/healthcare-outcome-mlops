import os 
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where() #secure connection to mongodb // certificate authority

