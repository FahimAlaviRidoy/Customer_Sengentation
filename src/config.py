import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Online Retail.xlsx')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
