import os
import numpy as np

# GCS PATHES
BUCKET_NAME = "arthur_tmp"
BASE_GCS_PATH = f"gs://{BUCKET_NAME}/analytics_cup"
PROCESSED_DIR = f"{BASE_GCS_PATH}/processed"
CLUSTERING_DIR = f"{BASE_GCS_PATH}/clustering"
RESULTS_DIR = f"{BASE_GCS_PATH}/results"
SOCCERMAP_MODELS_DIR = f"{BASE_GCS_PATH}/soccermap_model"

# LOCAL PATHES
MODELS_DIR = "models"

X_MIN, X_MAX = -52, 52
Y_MIN, Y_MAX = -34, 34
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
GOAL_CENTER  = np.array([105.0, 34.0])
GOAL_Y_TOP    = 34 + 7.32/2
GOAL_Y_BOTTOM = 34 - 7.32/2
GOAL_X        = 105.0

PITCH_CONTROL_PARAMS = {
    'v_max': 5.0,
    'a_max': 7.0,
    'reaction_time': 0.7,
    'lambda_param': 3.0,
    'time_max': 6.0,
    'dt': 0.02
}

PLAYER_POSITION_MAPPING = {
    "center_forward" : "Center Forward"
}

CLUSTERS_MAPPING = {
    'center_forward_low_block':
    {
        0:0,
        1:1,
        2:2,
        3:1,
        4:3,
        5:2,
        6:4,
        7:4,
        8:3,
        9:2,
        10:2
    }
}