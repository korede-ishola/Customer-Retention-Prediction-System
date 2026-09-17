from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn-Raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn-Processed.csv"

CATEGORICAL_FEATURES = ['gender', 'seniorcitizen', 'partner',
                        'dependents', 'phoneservice', 'multiplelines',
                        'internetservice', 'onlinesecurity', 'onlinebackup',
                        'deviceprotection', 'techsupport', 'streamingtv',
                        'streamingmovies', 'contract', 'paperlessbilling',
                        'paymentmethod']
NUMERICAL_FEATURES = ['tenure', 'monthlycharges', 'totalcharges']
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET = "churn"
TEST_SIZE = 0.15
THRESHOLD = 0.3
RANDOM_STATE = 42
FN_FP_COST_RATIO = 6

EXPERIMENT_FOLDER = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Telco Churn"

MODEL_ID = 'm-ad967ed3132048eaa8a125b8fa2723f1'
MODEL_NAME = "telco_churn_pipeline.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME