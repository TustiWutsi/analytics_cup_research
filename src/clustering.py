import os
import glob
import fsspec
import gcsfs
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import umap
import hdbscan

from .config import *
from .utils import *

def prepare_clustering_features_all_matches(
    player_position: str, 
    game_situation: tuple, 
    save_load_method: str = "gcp"
):
    """
    Transforms the raw results into a feature matrix suitable for clustering.
    Reads contextual information (player, team, minute, etc.) directly from 
    the processed match files (POSSESSION_DIR/<match_id>.parquet).

    Frames containing critical NaNs in 'player_position', 'distance_to_nearest_teammate',
    or 'distance_to_nearest_opponent' are excluded from the results to ensure data quality.

    The resulting metadata DataFrame includes:
        - match_id
        - player_id
        - player_name
        - player_team
        - opponent_team
        - minute

    Args:
        player_position (str): The target player position group.
        game_situation (tuple): The specific game situation filter (column, value).
        save_load_method (str): 'local' or 'gcp'.

    Returns:
        tuple: 
            - X (np.ndarray): The feature matrix (n_samples, n_features).
            - meta (pd.DataFrame): Metadata dataframe aligned with X.
            - frames_inputs (list): List of raw dictionary inputs for reference.
    """

    feature_vectors = []
    meta_info = []
    frames_inputs = []
    
    sub_dir = f"pitch_control_{player_position}_{game_situation[1]}"
    
    # --- 1. List files based on method ---
    if save_load_method == "gcp":
        fs = gcsfs.GCSFileSystem()
        PITCH_CONTROL_DIR = f"{BASE_GCS_PATH}/{sub_dir}"
        pitch_control_files = fs.glob(f"{PITCH_CONTROL_DIR}/*.npz")
    elif save_load_method == "local":
        PITCH_CONTROL_DIR = f"{BASE_LOCAL_PATH}/{sub_dir}"
        # using pathlib for robustness, converting to string
        pitch_control_files = [str(p) for p in Path(PITCH_CONTROL_DIR).glob("*.npz")]

    if not pitch_control_files:
        print(f"No files found in {PITCH_CONTROL_DIR}")
        return None, None, None

    # --- 2. Iterate and Process ---
    for file_path in tqdm(sorted(pitch_control_files)):
        # Extract match_id relative to the path
        if save_load_method == "gcp":
            match_id = file_path.split("/")[-1].replace(".npz", "")
        else:
            match_id = Path(file_path).stem

        # --- Read the .npz file ---
        try:
            if save_load_method == "gcp":
                with fs.open(file_path, "rb") as f:
                    data = np.load(f, allow_pickle=True)
                    results = data["results"].tolist()
            elif save_load_method == "local":
                data = np.load(file_path, allow_pickle=True)
                results = data["results"].tolist()
        except Exception as e:
            print(f"Error reading {file_path} : {e}")
            continue

        if not results:
            continue

        # --- Read the corresponding POSSESSION file ---
        try:
            if save_load_method == "gcp":
                possession_path = f"{PROCESSED_DIR}/{match_id}.parquet"
                with fs.open(possession_path, "rb") as fp:
                    possession_df = pd.read_parquet(fp)
            elif save_load_method == "local":
                possession_path = f"{PROCESSED_DIR_LOCAL}/{match_id}.parquet"
                possession_df = pd.read_parquet(possession_path)
        except Exception as e:
            print(f"Unable to read {possession_path} : {e}")
            possession_df = None

        # --- Prepare team names (to deduce opponent) ---
        all_teams = (
            possession_df["team_name"].unique().tolist()
            if possession_df is not None and "team_name" in possession_df.columns
            else []
        )

        # --- Iterate over valid results ---
        for item in results:
            # Exclusion if critical NaNs exist
            player_position_coords = item.get("player_position", (np.nan, np.nan))
            if (
                np.isnan(player_position_coords[0]) or np.isnan(player_position_coords[1]) or
                np.isnan(item.get("distance_to_nearest_teammate", np.nan)) or
                np.isnan(item.get("distance_to_nearest_opponent", np.nan))
            ):
                continue  # Skip this frame

            player_id = item.get("player_id")
            frame = item.get("frame")

            # (1) Flatten the Pitch Control map
            pitch_flat = item["pitch_control_map"].flatten()
            
            # Remove outliers based on pitch control coverage
            count_gt_05 = np.sum(pitch_flat > 0.5)
            if count_gt_05 > 70:
                continue

            # (2) Contextual features
            defensive_lines = np.array(item["defensive_lines"], dtype=float)
            defensive_lines = np.pad(
                defensive_lines, (0, 3 - len(defensive_lines)),
                mode='constant', constant_values=np.nan
            )

            ball_x, ball_y = item["ball_position"]
            in_possession = 1.0 if item["in_possession"] else 0.0

            context_features = np.array([
                *defensive_lines,
                ball_x,
                ball_y,
                in_possession,
                item["distance_to_ball"],
                item["distance_to_nearest_teammate"],
                item["distance_to_nearest_opponent"]
            ], dtype=float)

            full_vector = np.concatenate([pitch_flat, context_features])
            feature_vectors.append(full_vector)
            frames_inputs.append(item)

            # (3) Retrieve info from the possession parquet
            if possession_df is not None:
                row = possession_df[
                    (possession_df["frame"] == frame) &
                    (possession_df["player_id"] == player_id)
                ]
                if not row.empty:
                    row = row.iloc[0]
                    player_name = row.get("player_short_name", None)
                    player_team = row.get("team_name", None)
                    
                    time_s = row.get("time_s", None)
                    minute = int(time_s // 60) if time_s is not None else None

                    # Deduce opponent
                    opponent_team = None
                    if all_teams and player_team in all_teams:
                        op_candidates = [t for t in all_teams if t != player_team]
                        opponent_team = op_candidates[0] if op_candidates else None
                else:
                    player_name = None
                    player_team = None
                    opponent_team = None
                    minute = None
            else:
                player_name = None
                player_team = None
                opponent_team = None
                minute = None

            # (4) Add to meta_info
            meta_info.append({
                "match_id": match_id,
                "frame": frame,
                "player_id": player_id,
                "player_position_role": item.get("player_position_role", None),
                "player_name": player_name,
                "player_team": player_team,
                "opponent_team": opponent_team,
                "minute": minute,
                "game_situation": game_situation[1]
            })

    if not feature_vectors:
        print("No feature vectors generated (files may be empty)")
        return None, None, None

    # --- Final feature matrix ---
    X = np.vstack(feature_vectors)

    # --- Handle remaining NaNs (replace with column mean) ---
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    meta = pd.DataFrame(meta_info)
    return X, meta, frames_inputs

def prepare_clustering_features_single_match(
    results: list,
    processed_df: pd.DataFrame,
    match_id: str,
    game_situation: tuple
):
    """
    Transforms raw pitch control results into a feature matrix for clustering.

    This function processes data for a single match. It filters invalid frames,
    flattens pitch control maps, and extracts contextual metadata.

    Args:
        results (list): List of dictionaries containing pitch control results (from .npz).
        processed_df (pd.DataFrame): DataFrame containing tracking/possession data.
        match_id (str): The identifier of the match being processed.
        game_situation (tuple): The specific game situation (column, value).

    Returns:
        tuple:
            - X (np.ndarray): The feature matrix (n_samples, n_features).
            - meta (pd.DataFrame): Metadata dataframe aligned with X.
    """
    feature_vectors = []
    meta_info = []

    # --- Prepare team names (for opponent deduction) ---
    all_teams = (
        processed_df["team_name"].unique().tolist()
        if processed_df is not None and "team_name" in processed_df.columns
        else []
    )

    if not results:
        return None, None, None

    # --- Iterate over valid results ---
    for item in results:
        # Exclusion if critical NaNs exist
        player_position_coords = item.get("player_position", (np.nan, np.nan))
        if (
            np.isnan(player_position_coords[0]) or np.isnan(player_position_coords[1]) or
            np.isnan(item.get("distance_to_nearest_teammate", np.nan)) or
            np.isnan(item.get("distance_to_nearest_opponent", np.nan))
        ):
            continue  # Skip this frame

        player_id = item.get("player_id")
        frame = item.get("frame")

        # (1) Flatten the Pitch Control map
        pitch_flat = item["pitch_control_map"].flatten()
        
        # Remove outliers based on pitch control coverage
        count_gt_05 = np.sum(pitch_flat > 0.5)
        if count_gt_05 > 70:
            continue

        # (2) Contextual features
        defensive_lines = np.array(item["defensive_lines"], dtype=float)
        defensive_lines = np.pad(
            defensive_lines, (0, 3 - len(defensive_lines)),
            mode='constant', constant_values=np.nan
        )

        ball_x, ball_y = item["ball_position"]
        in_possession = 1.0 if item["in_possession"] else 0.0

        context_features = np.array([
            *defensive_lines,
            ball_x,
            ball_y,
            in_possession,
            item["distance_to_ball"],
            item["distance_to_nearest_teammate"],
            item["distance_to_nearest_opponent"]
        ], dtype=float)

        full_vector = np.concatenate([pitch_flat, context_features])
        feature_vectors.append(full_vector)

        # (3) Retrieve info from the processed DataFrame
        if processed_df is not None:
            row = processed_df[
                (processed_df["frame"] == frame) &
                (processed_df["player_id"] == player_id)
            ]
            if not row.empty:
                row = row.iloc[0]
                player_name = row.get("player_short_name", None)
                player_team = row.get("team_name", None)
                
                time_s = row.get("time_s", None)
                minute = int(time_s // 60) if time_s is not None else None

                # Deduce opponent
                opponent_team = None
                if all_teams and player_team in all_teams:
                    op_candidates = [t for t in all_teams if t != player_team]
                    opponent_team = op_candidates[0] if op_candidates else None
            else:
                player_name = None
                player_team = None
                opponent_team = None
                minute = None
        else:
            player_name = None
            player_team = None
            opponent_team = None
            minute = None

        # (4) Add to meta_info
        meta_info.append({
            "match_id": match_id,
            "frame": frame,
            "player_id": player_id,
            "player_position_role": item.get("player_position_role", None),
            "player_name": player_name,
            "player_team": player_team,
            "opponent_team": opponent_team,
            "minute": minute,
            "game_situation": game_situation[1]
        })

    if not feature_vectors:
        return None, None

    # --- Final feature matrix ---
    X = np.vstack(feature_vectors)

    # --- Handle remaining NaNs (replace with column mean) ---
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    meta = pd.DataFrame(meta_info)
    return X, meta

def train_pca_kmeans_clustering(
    X, 
    pca_components, 
    n_clusters, 
    dim=(32,50), 
    save: bool = False, 
    save_load_method: str = "gcp",
    player_position: str = "",
    game_situation_value: str = ""
):
    N_SPATIAL = dim[0]*dim[1]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "spatial",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_components, random_state=42)),
                ]),
                slice(0, N_SPATIAL),
            ),
            (
                "context",
                StandardScaler(),
                slice(N_SPATIAL, None),
            ),
        ]
    )
    
    clustering_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            ),
        ]
    )
    
    clustering_pipeline.fit(X)
    
    if save:
        filename = f"clustering_pipeline_{player_position}_{game_situation_value}.joblib"
        
        if save_load_method == "gcp":
            fs = gcsfs.GCSFileSystem()
            PIPELINE_PATH = f"{CLUSTERING_DIR}/{filename}"
            save_pipeline_gcs(
                fs=fs,
                gcs_path=PIPELINE_PATH,
                pipeline=clustering_pipeline,
            )
        elif save_load_method == "local":
            PIPELINE_PATH_LOCAL = f"{MODELS_DIR_LOCAL}/{filename}"
            os.makedirs(os.path.dirname(PIPELINE_PATH_LOCAL), exist_ok=True)
            joblib.dump(clustering_pipeline, PIPELINE_PATH_LOCAL)
        
    return clustering_pipeline

def predict_clusters(
    player_position: str, 
    game_situation: tuple, 
    X, 
    meta, 
    save_load_method: str = "gcp",
    cluster_mapping: dict = CLUSTERS_MAPPING
):
    """
    Predicts cluster labels for a feature matrix X using a pre-trained pipeline.
    Handles loading the model from either GCS or a local directory.

    Args:
        player_position (str): The target player position group.
        game_situation (tuple): The specific game situation tuple (column, value).
        X (np.ndarray): The feature matrix input for the model.
        meta (pd.DataFrame): The metadata DataFrame to enrich.
        save_load_method (str): 'local' or 'gcp' for data I/O.
        cluster_mapping (dict): Optional mapping to group raw clusters into semantic labels.

    Returns:
        pd.DataFrame: The metadata DataFrame updated with 'cluster' and 'cluster_gathered' columns.
    """
    situation_value = game_situation[1]
    mapping_key = f"{player_position}_{situation_value}"
    
    # We define fs here for GCS usage
    fs = gcsfs.GCSFileSystem()

    if save_load_method == "gcp":
        # --- GCP Loading ---
        PIPELINE_PATH_GCS = f"{CLUSTERING_DIR}/clustering_pipeline_{player_position}_{situation_value}.joblib"
        
        if fs.exists(PIPELINE_PATH_GCS):
            try:
                pipeline = load_pipeline_gcs(fs, PIPELINE_PATH_GCS)
                labels = pipeline.predict(X)
                meta["cluster"] = labels
                
                if cluster_mapping and mapping_key in cluster_mapping:
                    meta['cluster_gathered'] = meta['cluster'].map(cluster_mapping[mapping_key])
                else:
                    meta['cluster_gathered'] = meta['cluster']
            except Exception as e:
                print(f"Error loading/predicting from GCS: {e}")
                meta["cluster"] = None
                meta['cluster_gathered'] = None
        else:
            print(f"Pipeline not found on GCS: {PIPELINE_PATH_GCS}")
            meta["cluster"] = None
            meta['cluster_gathered'] = None
            
    elif save_load_method == "local":
        # --- Local Loading ---
        PIPELINE_PATH_LOCAL = f"{MODELS_DIR_LOCAL}/{f'clustering_pipeline_{player_position}_{situation_value}.joblib'}"
        
        if os.path.exists(PIPELINE_PATH_LOCAL):
            try:
                # Assuming load_pipeline_local is just joblib.load or similar wrapper
                pipeline = joblib.load(PIPELINE_PATH_LOCAL) 
                labels = pipeline.predict(X)
                meta["cluster"] = labels
                
                if cluster_mapping and mapping_key in cluster_mapping:
                    meta['cluster_gathered'] = meta['cluster'].map(cluster_mapping[mapping_key])
                else:
                    meta['cluster_gathered'] = meta['cluster']
            except Exception as e:
                print(f"Error loading/predicting locally: {e}")
                meta["cluster"] = None
                meta['cluster_gathered'] = None
        else:
            print(f"Pipeline not found locally: {PIPELINE_PATH_LOCAL}")
            meta["cluster"] = None
            meta['cluster_gathered'] = None

    return meta

def plot_cluster_summary_pitch_control(
    results: list,
    meta: pd.DataFrame,
    n_cols: int = 5,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    half_pitch: bool = True
):
    """
    Displays the mean pitch control map and average contextual features 
    for specific clusters, including global information based on 'cluster_gathered'.

    Args:
        results (list): List of dictionaries containing pitch control results and features.
        meta (pd.DataFrame): Metadata dataframe containing cluster labels.
        n_cols (int): Number of columns for the subplot grid.
        pitch_length (float): Length of the pitch in meters.
        pitch_width (float): Width of the pitch in meters.
        half_pitch (bool): If True, displays only the attacking half of the pitch.
    """

    # --- Clusters to display ---
    clusters_to_display = [0, 1, 8, 7, 5]
    meta = meta.copy()

    n_clusters = len(clusters_to_display)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="white")
    pcm = None

    total_frames = len(meta)  # Total frames across the dataset

    for i, cluster_id in enumerate(clusters_to_display):
        ax = axes[i]

        # ---- Frames for current cluster ----
        cluster_indices = meta.index[meta["cluster"] == cluster_id].tolist()
        cluster_items = [results[idx] for idx in cluster_indices]

        if len(cluster_items) == 0:
            ax.axis("off")
            continue

        # ---- Mean pitch control maps ----
        # 
        maps = np.array([item["pitch_control_map"] for item in cluster_items])
        mean_map = np.nanmean(maps, axis=0)

        # ---- Mean defensive lines ----
        def_lines_list = []
        for item in cluster_items:
            lines = np.array(item.get("defensive_lines", []), dtype=float)
            if lines.size == 0:
                padded = np.array([np.nan, np.nan, np.nan])
            else:
                padded = np.pad(lines, (0, max(0, 3 - len(lines))), mode="constant", constant_values=np.nan)
            def_lines_list.append(padded)
        all_lines = np.vstack(def_lines_list)
        mean_lines = np.nanmean(all_lines, axis=0)

        # ---- Mean contextual features ----
        def nanmean_safe(lst):
            arr = np.array(lst, dtype=float)
            if np.all(np.isnan(arr)):
                return np.nan
            return float(np.nanmean(arr))

        mean_dist_ball = nanmean_safe([item.get("distance_to_ball", np.nan) for item in cluster_items])
        mean_dist_tm = nanmean_safe([item.get("distance_to_nearest_teammate", np.nan) for item in cluster_items])
        mean_dist_op = nanmean_safe([item.get("distance_to_nearest_opponent", np.nan) for item in cluster_items])

        in_poss_array = np.array([1.0 if item.get("in_possession", False) else 0.0 for item in cluster_items])
        possession_ratio = float(np.nanmean(in_poss_array)) if in_poss_array.size > 0 else np.nan

        # ---- Total frames in gathered cluster ----
        cluster_gathered_id = meta.loc[meta["cluster"] == cluster_id, "cluster_gathered"].iloc[0]
        total_frames_gathered = (meta["cluster_gathered"] == cluster_gathered_id).sum()
        ratio_frames = total_frames_gathered / total_frames

        # ---- Pitch creation ----
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            line_color="black",
            pitch_color="white"
        )

        H, W = mean_map.shape
        x = np.linspace(0, pitch_length, W)
        y = np.linspace(0, pitch_width, H)
        bin_statistic = dict(statistic=mean_map, x_grid=x, y_grid=y)
        
        cmap = plt.cm.RdBu_r
        cmap.set_bad(color="white")

        pitch.draw(ax=ax)
        pcm = pitch.heatmap(
            bin_statistic,
            ax=ax,
            cmap='viridis',
            vmin=0,
            vmax=1,
            alpha=0.9,
        )

        if half_pitch:
            ax.set_xlim(pitch_length / 2, pitch_length)
            ax.set_ylim(0, pitch_width)

        # ---- Defensive lines ----
        if mean_lines.size > 0 and not np.all(np.isnan(mean_lines)):
            colors = ["black", "gray", "silver"]
            for j, x_mean in enumerate(mean_lines):
                if not np.isnan(x_mean):
                    ax.axvline(x=float(x_mean), color=colors[j % len(colors)], linestyle="--", linewidth=2)

        # ---- Text formatting ----
        def fmt(v):
            return f"{v:.2f}" if (v is not None and not np.isnan(v)) else "n/a"

        stats_text = (
            f"Share of total frames : ({ratio_frames:.1%})\n"
            f"Possession ratio : {possession_ratio:.0%}\n"
            f"Avg ball distance : {fmt(mean_dist_ball)} m\n"
            f"Avg closest teammate distance : {fmt(mean_dist_tm)} m\n"
            f"Avg closest opponent distance : {fmt(mean_dist_op)} m"
        )

        ax.set_title(f"Cluster {i}", fontsize=17, fontweight="bold")
        ax.text(0.5, -0.05, stats_text, transform=ax.transAxes,
                fontsize=17, ha="center", va="top", linespacing=1.6)

    # ---- Remove empty axes ----
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(right=0.92, wspace=0.35, hspace=0.6)

    # ---- Global colorbar ----
    if pcm is not None:
        cbar_ax = fig.add_axes([0.94, 0.05, 0.01, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label("Mean Pitch Control Probability", fontsize=11)
        
    #plt.savefig(
    #    "images/ipc_clusters_center_forward_low_block.png",
    #    dpi=300,               
    #    bbox_inches='tight',    
    #    facecolor='white'
    #)

    plt.show()
    
def plot_players_cluster_distribution(
    meta: pd.DataFrame, 
    n_players: int = 20
):
    """
    Visualizes the proportional distribution of clusters for the most frequent players.
    
    Generates a stacked horizontal bar chart showing how different clusters are 
    distributed for the top `n_players` found in the metadata.

    Args:
        meta (pd.DataFrame): The metadata dataframe containing 'player_name' and 'cluster_gathered'.
        n_players (int): The number of top players to display (based on frame count).
        title (str): The title of the plot.
    """
    
    # Identify top players based on frame count
    player_counts = meta['player_name'].value_counts().head(n_players)
    top_players = player_counts.index

    # Filter the DataFrame to keep only these top players
    df_top = meta[meta['player_name'].isin(top_players)]

    # Calculate cluster distribution per player
    cluster_distribution = (
        df_top.groupby(['player_name', 'cluster_gathered'])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to proportions for each player
    cluster_distribution = cluster_distribution.div(cluster_distribution.sum(axis=1), axis=0)

    # Plotting
    ax = cluster_distribution.plot(
        kind='barh',
        stacked=True,
        figsize=(12, 6),
        colormap='viridis',
        edgecolor='black'
    )
    
    player_position = meta.player_position_role.iloc[0]
    game_situation = meta.game_situation.iloc[0]
    plt.title(f"Clusters distribution for {player_position} in {game_situation} situation", fontsize=14, weight='bold')
    plt.xlabel("Proportion of Frames")
    plt.ylabel("Player")

    # Clean legend
    plt.legend(
        title="Cluster",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()
    
    #plt.savefig(
    #    "images/ipc_clusters_distribution_center_forward_low_block.png",
    #    dpi=300,               
    #    bbox_inches='tight',    
    #    facecolor='white'
    #)
    
    plt.show()