BUCKET_NAME = "arthur_tmp"
BASE_GCS_PATH = f"gs://{BUCKET_NAME}/analytics_cup"
PROCESSED_DIR = f"{BASE_GCS_PATH}/processed"
CLUSTERING_DIR = f"{BASE_GCS_PATH}/clustering"
RESULTS_DIR = f"{BASE_GCS_PATH}/results"
SOCCERMAP_MODELS_DIR = f"{BASE_GCS_PATH}/soccermap_model"

X_MIN, X_MAX = -52, 52
Y_MIN, Y_MAX = -34, 34
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

PITCH_CONTROL_PARAMS = {
    'v_max': 5.0,
    'a_max': 7.0,
    'reaction_time': 0.7,
    'lambda_param': 3.0,
    'time_max': 6.0,
    'dt': 0.02
}