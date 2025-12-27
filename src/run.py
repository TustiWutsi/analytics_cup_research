import fsspec
import gcsfs
from .individual_pitch_control import process_all_matches_parallel
from .clustering import prepare_clustering_features_all_matches, predict_clusters
from .soccermap_models import compute_metrics

from .config import *
from .utils import *

def run_all(
    player_position: str,
    game_situation: tuple,
    max_workers: int = 10,
    batch_size: int = 32
):
    """
    """
    
    fs = gcsfs.GCSFileSystem()
    parquet_files = fs.glob(f"{PROCESSED_DIR}/*.parquet")

    print(f"{len(parquet_files)} matchs détectés")
    
    ########### INDIVIDUAL PITCH CONTROL ###########
    print("1) Calculate Individual Pitch Control")

    process_all_matches_parallel(
        player_position=player_position,
        game_situation=game_situation,
        pitch_control_resolution= 1,
        max_workers= 10,
    )

    ########### SPACES CLUSTERING ###########
    print("2) Predict spaces clusters if a clustering model is available")
    
    X, meta, frames_inputs = prepare_clustering_features_all_matches(player_position, game_situation)
    
    meta = predict_clusters(
        player_position=player_position, 
        game_situation=game_situation, 
        X=X, 
        meta=meta, 
        model_local=False
    )
        
    ########### CALCULATE iscT METRICS ###########
    print("3) Run xPass and xT predictions and calculate iscT metrics")
        
    meta_out = compute_metrics(
        meta=meta,
        model_local=False,
        batch_size=32,
        save_to_gcs=True
    )
    
    return meta_out