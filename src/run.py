import fsspec
import gcsfs
import glob
from pathlib import Path
from .individual_pitch_control import process_all_matches_parallel
from .clustering import prepare_clustering_features_all_matches, predict_clusters
from .soccermap_models import compute_metrics

from .config import *
from .utils import *

def run_all(
    player_position: str,
    game_situation: tuple,
    max_workers: int = 10,
    batch_size: int = 32,
    save: bool = True,
    save_load_method: str = "gcp"
):
    """
    Orchestrates the full pipeline:
    1. Calculates Individual Pitch Control for all matches.
    2. Generates clustering features and predicts cluster labels.
    3. Computes iscT metrics using xPass and xThreat models.

    Args:
        player_position (str): Targeted player position.
        game_situation (tuple): Game situation filter (column, value).
        max_workers (int): Parallel workers for pitch control calculation.
        batch_size (int): Batch size for metrics computation.
        save (bool): Whether to save results at each step.
        save_load_method (str): 'local' or 'gcp'.
    """
    
    if save_load_method == "gcp":
        fs = gcsfs.GCSFileSystem()
        parquet_files = fs.glob(f"{PROCESSED_DIR}/*.parquet")
    elif save_load_method == "local":
        parquet_files = glob.glob(f"{PROCESSED_DIR_LOCAL}/*.parquet")
        
    print(f"{len(parquet_files)} matches detected")
    
    ########### INDIVIDUAL PITCH CONTROL ###########
    print("1) Calculate Individual Pitch Control")

    process_all_matches_parallel(
        player_position=player_position,
        game_situation=game_situation,
        pitch_control_resolution=1,
        max_workers=max_workers,
        save=save,
        save_load_method=save_load_method
    )

    ########### SPACES CLUSTERING ###########
    print("2) Predict spaces clusters if a clustering model is available")
    
    # Prepare features
    X, meta, frames_inputs = prepare_clustering_features_all_matches(
        player_position=player_position, 
        game_situation=game_situation,
        save_load_method=save_load_method
    )
    
    if X is None or meta is None:
        print("Skipping clustering and metrics due to lack of data.")
        return None

    # Predict clusters
    meta = predict_clusters(
        player_position=player_position, 
        game_situation=game_situation, 
        X=X, 
        meta=meta, 
        save_load_method=save_load_method
    )
        
    ########### CALCULATE iscT METRICS ###########
    print("3) Run xPass and xT predictions and calculate iscT metrics")
        
    meta_out = compute_metrics(
        meta=meta,
        batch_size=batch_size,
        save=save,
        save_load_method=save_load_method
    )
    
    return meta_out