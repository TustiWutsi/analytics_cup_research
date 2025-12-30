import pandas as pd
import json
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
import concurrent.futures
import fsspec
import gcsfs
from functools import partial
import glob

from .config import *
from .utils import *


def get_player_id_position(tracking_df: pd.DataFrame, player_position: str) -> dict:
    """
    Returns the IDs of players occupying a specific position for each team.
    If no player is found for a team, the corresponding value is None.

    Args:
        tracking_df (pd.DataFrame): The dataframe containing tracking data.
        player_position (str): The specific position group to filter by.

    Returns:
        dict: A dictionary mapping team IDs to player IDs (or None).
    """
    home_team_id = tracking_df["home_team_id"].iloc[0]
    away_team_id = tracking_df["away_team_id"].iloc[0]

    def get_player_id(team_id):
        filtered = tracking_df[
            (tracking_df["position_group"] == player_position)
            & (tracking_df["team_id"] == team_id)
        ]
        return filtered["player_id"].iloc[0] if not filtered.empty else None

    return {
        home_team_id: get_player_id(home_team_id),
        away_team_id: get_player_id(away_team_id),
    }

def build_team_tracking(df: pd.DataFrame, team_side: str = 'home'):
    """
    Constructs a tracking dataset in a format similar to Metrica: 
    one row per frame, with positions and velocities for each player of the team.

    Args:
        df (pd.DataFrame): The raw dataframe.
        team_side (str): The side of the team ('home' or 'away').

    Returns:
        tuple: A tuple containing the formatted tracking DataFrame and a list of player IDs.
    """
    team_id = df[f'{team_side}_team_id'].iloc[0]
    df_team = df[df['team_id'] == team_id].copy()
    team_name = df_team['team_name'].iloc[0]
    
    # List of players
    players = df_team['player_id'].unique()
    
    tracking = (
        df_team
        .pivot(index='frame', columns='player_id', values=['x_rescaled', 'y_rescaled'])
        .sort_index()
    )
    tracking.columns = [f"{c[0]}_{int(c[1])}" for c in tracking.columns]
    tracking = tracking.reset_index()

    # Add ball data
    ball = (
        df[df['is_ball']]
        .set_index('frame')[['x_rescaled', 'y_rescaled']]
        .rename(columns={'x_rescaled': 'ball_x', 'y_rescaled': 'ball_y'})
    )
    tracking = tracking.merge(ball, on='frame', how='left')
    
    tracking['player_id_in_possession'] = df.groupby('frame')['player_in_possession_id'].first().values
    tracking['Period'] = df.groupby('frame')['period'].first().values
    tracking['Time [s]'] = df.groupby('frame')['time'].first().values
    tracking['team_name'] = team_name
    
    return tracking, players

def calculate_player_velocities(tracking: pd.DataFrame, players: list, frame_rate: int = 4) -> pd.DataFrame:
    """
    Calculates velocity components (vx, vy) for each player in meters/second.

    Args:
        tracking (pd.DataFrame): The tracking dataframe.
        players (list): List of player IDs to process.
        frame_rate (int): The number of frames per second.

    Returns:
        pd.DataFrame: The tracking dataframe with added velocity columns.
    """
    for pid in players:
        for axis in ['x', 'y']:
            col = f"{axis}_rescaled_{pid}"
            vcol = f"v{axis}_{pid}"
            tracking[vcol] = tracking[col].diff() * frame_rate
    return tracking

def time_to_intercept(player: dict, target: tuple, params: dict) -> float:
    """
    Calculates the minimum time for a `player` to reach a `target`, considering:
    - Projection of velocity onto the target direction (direction),
    - Maximum acceleration (a_max),
    - Maximum velocity (v_max),
    - Reaction time (reaction_time).

    Args:
        player (dict): Dictionary with keys 'x', 'y', 'vx', 'vy'.
        target (tuple): Target coordinates (tx, ty).
        params (dict): Dictionary containing 'v_max', 'a_max', 'reaction_time'.

    Returns:
        float: The interception time (t_intercept) in seconds.
    """
    v_max = params['v_max']
    a_max = params['a_max']
    reaction_time = params['reaction_time']

    px, py = float(player['x']), float(player['y'])
    vx, vy = float(player.get('vx', 0.0)), float(player.get('vy', 0.0))
    tx, ty = float(target[0]), float(target[1])

    dx = tx - px
    dy = ty - py
    d = np.hypot(dx, dy)
    if d < 1e-6:
        return reaction_time  # Already practically at location

    # Unit direction towards target
    ux, uy = dx / d, dy / d

    # Projection of velocity onto target direction
    v_proj = vx * ux + vy * uy
    v_proj = max(0.0, v_proj)  # If moving away, immediate contribution is 0

    # Assume that after reaction time, the player accelerates at a_max towards the target
    # with initial velocity v_proj (projection). They cannot exceed v_max.
    # Calculate time needed to cover distance d after reaction_time.

    # CASE 1: If initial velocity is already >= v_max (rare), cover d at v_proj (capped at v_max)
    v0 = min(v_proj, v_max)

    # Fallback if acceleration is zero
    if a_max <= 1e-6:
        motion_time = d / max(1e-6, v0 if v0 > 1e-6 else v_max)
        return reaction_time + motion_time

    # Time to accelerate from v0 to v_max
    if v0 < v_max:
        t_acc = (v_max - v0) / a_max
        # Distance covered during acceleration
        dist_acc = v0 * t_acc + 0.5 * a_max * t_acc**2
    else:
        t_acc = 0.0
        dist_acc = 0.0

    # If distance to target is less than dist_acc,
    # the player does not reach v_max before arriving. Solve quadratic:
    # 0.5 * a * t^2 + v0 * t - d = 0  (t >= 0)
    if dist_acc >= d and a_max > 0:
        # solve 0.5 a t^2 + v0 t - d = 0
        A = 0.5 * a_max
        B = v0
        C = -d
        disc = B*B - 4*A*C
        if disc < 0:
            disc = 0.0
        t_motion = (-B + np.sqrt(disc)) / (2*A)
        motion_time = t_motion
    else:
        # Reaches v_max (potentially), and covers the rest at v_max
        dist_rem = max(0.0, d - dist_acc)
        if v_max <= 1e-6:
            # Degenerate fallback
            motion_time = t_acc + dist_rem / (v0 + 1e-6)
        else:
            motion_time = t_acc + dist_rem / v_max

    return reaction_time + motion_time

def calculate_pitch_control_at_target_full(target_pos, attacking_players, defending_players, params):
    """
    Full version inspired by 'Physics of Possession'.
    - Computes t_i for each player (direction + accel + v_max + reaction).
    - Discretizes time t = 0..time_max with step dt.
    - At each step t, calculates weights w_i(t) = exp(-lambda*(t - t_i)) for t >= t_i (0 otherwise).
    - Normalizes weights among all players still 'available' at time t.
    - Distributes interception probability at this step to each player proportionally.
    - Accumulates probability per team.

    Args:
        target_pos (tuple): (x, y) coordinates of the target.
        attacking_players (list): List of attacking player states.
        defending_players (list): List of defending player states.
        params (dict): Simulation parameters.

    Returns:
        dict: Dictionary containing probabilities {'P_att': float, 'P_def': float}.
    """
    # Parameters
    lambda_param = params.get('lambda_param', 3.0)   # Decay rate of time weights
    time_max = params.get('time_max', 6.0)           # Horizon (s)
    dt = params.get('dt', 0.02)                      # Integration step (s)
    goalkeeper_ids = params.get('goalkeeper_ids', set())  # Handle GK differently if needed (optional)

    # Filter valid players
    att_valid = [p for p in attacking_players if (p is not None and not np.isnan(p['x']) and not np.isnan(p['y']))]
    def_valid = [p for p in defending_players if (p is not None and not np.isnan(p['x']) and not np.isnan(p['y']))]

    # Boundary cases
    if len(att_valid) == 0 and len(def_valid) == 0:
        return {'P_att': 0.5, 'P_def': 0.5}
    if len(att_valid) == 0:
        return {'P_att': 0.0, 'P_def': 1.0}
    if len(def_valid) == 0:
        return {'P_att': 1.0, 'P_def': 0.0}

    # Calculate mean arrival times (ti) for each player
    all_players = []
    for p in att_valid:
        ti = time_to_intercept(p, target_pos, params)
        all_players.append({'team': 'att', 'player': p, 't_mean': ti})
    for p in def_valid:
        ti = time_to_intercept(p, target_pos, params)
        all_players.append({'team': 'def', 'player': p, 't_mean': ti})

    # Optional sort (not necessary), we perform integration
    # Time discretization
    t_arr = np.arange(0.0, time_max + dt/2, dt)

    # Accumulated probabilities by player and team
    prob_player = np.zeros(len(all_players))  # Prob mass accumulated by player
    prob_team_att = 0.0
    prob_team_def = 0.0

    # Track total probability already captured (to stop if close to 1)
    captured_total = 0.0

    # For each time step, calculate weights and distribute small probability mass
    
    # Intuition: the earlier a player arrives (ti), the better, because we sum weights potentially over more t >= ti.
    # Intuition: but the further t gets from ti, the smaller the weight (exponential decay), so for t > ti of many players, the advantage of arriving first diminishes.
    for k, t in enumerate(t_arr):
        # Weights for each player at this time (0 if t < t_mean)
        weights = np.array([np.exp(-lambda_param * max(0.0, (t - pl['t_mean']))) if t >= pl['t_mean'] else 0.0 for pl in all_players])

        sumw = np.sum(weights)
        if sumw <= 0:
            continue  # Nothing to distribute at this step

        # Probability 'mass' available at this step: the portion not yet captured
        mass = (1.0 - captured_total)
        # We can weight mass by dt (for integration approximation).
        # Here we distribute mass * (lambda_norm * dt) or heuristic approximation.
        # For stability, we define a small fraction per step:
        small_mass = mass * (1 - np.exp(-lambda_param * dt))  # Heuristic -> approx mass * lambda*dt for small dt

        # Distribute small_mass proportionally to weights
        probs_at_t = small_mass * (weights / sumw)

        # Add to prob_player & teams
        prob_player += probs_at_t
        # Update captured_total
        captured_total = np.sum(prob_player)
        if captured_total >= 0.9999:
            break

    # Sum by team
    for idx, pl in enumerate(all_players):
        if pl['team'] == 'att':
            prob_team_att += prob_player[idx]
        else:
            prob_team_def += prob_player[idx]

    # Normalization (numerical safety)
    total = prob_team_att + prob_team_def
    if total <= 0:
        return {'P_att': 0.5, 'P_def': 0.5}
    P_att = prob_team_att / total
    P_def = prob_team_def / total

    return {'P_att': float(P_att), 'P_def': float(P_def)}

def get_players_state(tracking, frame, players):
    """
    Retrieves positions and velocities of players for a given frame, ignoring NaNs.

    Args:
        tracking (pd.DataFrame): The tracking data.
        frame (int): The specific frame number.
        players (list): List of player IDs.

    Returns:
        list: A list of dictionaries [{'id', 'x', 'y', 'vx', 'vy'}, ...].
    """
    row = tracking.loc[tracking['frame'] == frame]
    if row.empty:
        return []
    row = row.iloc[0]
    player_states = []
    for pid in players:
        x = row.get(f'x_rescaled_{pid}', np.nan)
        y = row.get(f'y_rescaled_{pid}', np.nan)
        vx = row.get(f'vx_{pid}', np.nan)
        vy = row.get(f'vy_{pid}', np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        player_states.append({
            'id': pid,
            'x': float(x),
            'y': float(y),
            'vx': 0.0 if pd.isna(vx) else float(vx),
            'vy': 0.0 if pd.isna(vy) else float(vy)
        })
    return player_states

def calculate_individual_pitch_control_all_pitch(
    player_id,
    frame,
    home_tracking,
    away_tracking,
    home_players,
    away_players,
    params,
    pitch_length=105,
    pitch_width=68,
    resolution=3,
    half_pitch=False
):
    """
    Calculates the Pitch Control for a specific player (player_id) 
    against all opponents at a given instant (frame).
    
    If half_pitch=True, the calculation covers only the right half of the pitch.

    Args:
        player_id (str): The ID of the focus player.
        frame (int): Frame number.
        home_tracking (pd.DataFrame): Home team tracking data.
        away_tracking (pd.DataFrame): Away team tracking data.
        home_players (list): List of home player IDs.
        away_players (list): List of away player IDs.
        params (dict): Simulation parameters.
        pitch_length (float): Length of the pitch.
        pitch_width (float): Width of the pitch.
        resolution (int): Grid resolution factor.
        half_pitch (bool): Whether to compute for half pitch only.

    Returns:
        np.ndarray: The 2D pitch control grid.
    """

    # ---- 1. Identify teams and players ----
    if player_id in home_players:
        att_players = get_players_state(home_tracking, frame, [player_id])
        def_players = get_players_state(away_tracking, frame, away_players)
    elif player_id in away_players:
        att_players = get_players_state(away_tracking, frame, [player_id])
        def_players = get_players_state(home_tracking, frame, home_players)
    else:
        raise ValueError(f"player_id {player_id} not found in tracking data")

    # ---- 2. Spatial grid ----
    n_grid_x = int(50 * resolution)
    n_grid_y = int(32 * resolution)

    if half_pitch:
        x_min, x_max = pitch_length / 2, pitch_length
    else:
        x_min, x_max = 0, pitch_length

    x = np.linspace(x_min, x_max, n_grid_x)
    y = np.linspace(0, pitch_width, n_grid_y)
    P_player_grid = np.zeros((len(y), len(x)))

    # ---- 3. Calculate pitch control on each cell ----
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            try:
                p = calculate_pitch_control_at_target_full(
                    (xx, yy), att_players, def_players, params
                )
                P_player_grid[iy, ix] = (
                    p["P_att"] if not np.isnan(p["P_att"]) else 0.5
                )
            except Exception:
                P_player_grid[iy, ix] = 0.5

    P_player_grid[P_player_grid == 0.5] = 0

    return P_player_grid

def calculate_defensive_lines(
    frame,
    away_tracking,
    away_players,
    n_clusters_options=(2, 3)
):
    """
    Calculates the x-coordinates of defensive lines using clustering on player positions.

    Args:
        frame (int): The current frame.
        away_tracking (pd.DataFrame): Tracking data for the opponent/away team.
        away_players (list): List of opponent player IDs.
        n_clusters_options (tuple): Options for number of clusters to test.

    Returns:
        list: Sorted x-coordinates of the defensive lines (or None if insufficient data).
    """

    # Extract (x, y) positions of opponent players for the frame
    positions = []
    player_ids = []
    row = away_tracking.loc[away_tracking['frame'] == frame]
    for pid in away_players:
        if f"x_rescaled_{pid}" in row.columns and f"y_rescaled_{pid}" in row.columns:
            x = row[f"x_rescaled_{pid}"].values[0]
            y = row[f"y_rescaled_{pid}"].values[0]
            if not np.isnan(x) and not np.isnan(y):
                positions.append([x, y])
                player_ids.append(pid)

    if len(positions) < 4:
        print(f"Frame {frame}: not enough opponent players to form defensive lines.")
        return None

    positions = np.array(positions)

    # Identify and remove the goalkeeper (the player with the lowest x value)
    gk_index = np.argmax(positions[:, 0])  # Closest to goal (assuming left-to-right or specific coords)
    positions = np.delete(positions, gk_index, axis=0)
    player_ids = np.delete(player_ids, gk_index)

    # Test multiple cluster numbers (typically 2 or 3)
    best_model = None
    best_score = np.inf

    for n_clusters in n_clusters_options:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(positions[:, [0]])  # Cluster based on x only
        # Measure intra-cluster cohesion (mean standard deviation)
        intra_var = np.mean([np.std(positions[labels == k, 0]) for k in range(n_clusters)])
        if intra_var < best_score:
            best_score = intra_var
            best_model = model

    # Retrieve final clusters
    labels = best_model.fit_predict(positions[:, [0]])
    n_clusters = len(np.unique(labels))

    # Calculate mean line per cluster
    line_means = []
    for k in range(n_clusters):
        cluster_pts = positions[labels == k]
        x_mean = np.mean(cluster_pts[:, 0])
        line_means.append(x_mean)

    # Sort lines from closest to goal to furthest
    line_means = sorted(line_means)

    return line_means

def extract_player_context(
    frame: int,
    player_id: str,
    home_tr: pd.DataFrame,
    away_tr: pd.DataFrame,
    home_players: list,
    away_players: list
) -> dict:
    """
    Extracts context information for a player at a given frame:
    - Player position
    - Ball position
    - Possession status (True/False)
    - Distance to ball
    - Distance to nearest teammate
    - Distance to nearest opponent

    Args:
        frame (int): Frame number.
        player_id (str): ID of the player.
        home_tr (pd.DataFrame): Home tracking data.
        away_tr (pd.DataFrame): Away tracking data.
        home_players (list): List of home player IDs.
        away_players (list): List of away player IDs.

    Returns:
        dict: Contextual metrics.
    """

    # Determine if player is home or away
    if player_id in home_players:
        team_tr, opp_tr = home_tr, away_tr
        teammates = [p for p in home_players if p != player_id]
        opponents = away_players
    elif player_id in away_players:
        team_tr, opp_tr = away_tr, home_tr
        teammates = [p for p in away_players if p != player_id]
        opponents = home_players
    else:
        raise ValueError(f"player_id {player_id} belongs to neither team.")

    # Extract row for the frame
    row_team = team_tr.loc[team_tr["frame"] == frame]
    row_opp = opp_tr.loc[opp_tr["frame"] == frame]
    if row_team.empty or row_opp.empty:
        raise ValueError(f"Frame {frame} missing from tracking data.")

    # ---- Player Position ----
    player_pos = np.array([
        row_team[f"x_rescaled_{player_id}"].values[0],
        row_team[f"y_rescaled_{player_id}"].values[0]
    ])

    # ---- Ball Position ----
    ball_pos = np.array([
        row_team["ball_x"].values[0],
        row_team["ball_y"].values[0]
    ])

    # ---- Is player in possession? ----
    player_in_possession_id = row_team["player_id_in_possession"].values[0]
    in_possession = player_in_possession_id == player_id

    # ---- Distance to ball ----
    distance_to_ball = np.linalg.norm(player_pos - ball_pos)

    # ---- Distance to nearest teammate ----
    teammate_positions = []
    for pid in teammates:
        x, y = row_team[f"x_rescaled_{pid}"].values[0], row_team[f"y_rescaled_{pid}"].values[0]
        if not np.isnan(x) and not np.isnan(y):
            teammate_positions.append([x, y])

    if teammate_positions:
        teammate_positions = np.array(teammate_positions)
        distance_to_nearest_teammate = np.min(np.linalg.norm(teammate_positions - player_pos, axis=1))
    else:
        distance_to_nearest_teammate = np.nan

    # ---- Distance to nearest opponent ----
    opponent_positions = []
    for pid in opponents:
        x, y = row_opp[f"x_rescaled_{pid}"].values[0], row_opp[f"y_rescaled_{pid}"].values[0]
        if not np.isnan(x) and not np.isnan(y):
            opponent_positions.append([x, y])

    if opponent_positions:
        opponent_positions = np.array(opponent_positions)
        distance_to_nearest_opponent = np.min(np.linalg.norm(opponent_positions - player_pos, axis=1))
    else:
        distance_to_nearest_opponent = np.nan

    # ---- Summary ----
    return {
        "frame": frame,
        "player_id": player_id,
        "player_position": tuple(player_pos),
        "ball_position": tuple(ball_pos),
        "in_possession": in_possession,
        "distance_to_ball": distance_to_ball,
        "distance_to_nearest_teammate": distance_to_nearest_teammate,
        "distance_to_nearest_opponent": distance_to_nearest_opponent
    }

def extract_player_pitch_control_and_contextual(
    player_id,
    frame,
    home_tracking,
    away_tracking,
    home_players,
    away_players,
    params,
    pitch_length=105,
    pitch_width=68,
    resolution=1,
    half_pitch=True
):
    """
    Combines individual pitch control calculation and contextual data extraction.
    
    Args:
        player_id (str): Focus player ID.
        frame (int): Frame number.
        home_tracking (pd.DataFrame): Home tracking data.
        away_tracking (pd.DataFrame): Away tracking data.
        home_players (list): Home player IDs.
        away_players (list): Away player IDs.
        params (dict): Simulation parameters.
        pitch_length (float): Pitch length.
        pitch_width (float): Pitch width.
        resolution (int): Resolution of the grid.
        half_pitch (bool): Flag for half-pitch calculation.

    Returns:
        dict: A dictionary containing 'pitch_control_map', 'defensive_lines', and context metrics.
    """

    # ---- 1. Individual Pitch Control Calculation ----
    P_player_grid = calculate_individual_pitch_control_all_pitch(
        player_id,
        frame,
        home_tracking,
        away_tracking,
        home_players,
        away_players,
        params,
        pitch_length,
        pitch_width,
        resolution,
        half_pitch
    )

    # ---- 2. Identify defensive lines ----
    line_means = calculate_defensive_lines(
        frame,
        away_tracking,
        away_players,
        n_clusters_options=(2, 3)
    )

    # ---- 3. Extract player context ----
    player_context = extract_player_context(
        frame,
        player_id,
        home_tracking,
        away_tracking,
        home_players,
        away_players
    )

    player_frame_info = {
        "pitch_control_map": P_player_grid,
        "defensive_lines": line_means
    }

    player_frame_info.update(player_context)

    return player_frame_info

def extract_all_for_position_across_match(
    match_id: str,
    player_position: str,
    game_situation: tuple,
    df=None,
    params: dict = PITCH_CONTROL_PARAMS,
    sample_size: int = None,
    resolution: int = 1,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
    save_load_method: str = "gcp"
):
    """
    Calculates pitch control maps and player context for a given position,
    for all frames corresponding to a specific game situation, across a whole match.

    Args:
        match_id (str): The unique identifier of the match.
        player_position (str): The position group to filter.
        game_situation (tuple): Condition to filter frames (col_name, value).
        df (pd.DataFrame): Processed tracking dataframe of a single match.
        params (dict): Pitch Control parameters.
        resolution (int): Grid resolution.
        pitch_length (float): Pitch length.
        pitch_width (float): Pitch width.
        save_load_method (str): 'local' or 'gcp'

    Returns:
        list: A list of dictionaries containing frame data, pitch_control_map, 
              defensive_lines, and contextual metrics.
    """

    all_results = []

    # ---- Load dataframe ----
    if df is None:
        if save_load_method == "gcp":
            df = pd.read_parquet(
                f"{PROCESSED_DIR}/{match_id}.parquet",
                storage_options={"token": "google_default"},
            )
        elif save_load_method == "local":
            df = pd.read_parquet(f"{PROCESSED_DIR_LOCAL}/{match_id}.parquet")

    # Filter on game situation
    df = df[df[game_situation[0]] == game_situation[1]]
    
    # ---- Identify player_ids for the position ----
    player_position_skillcorner = PLAYER_POSITION_MAPPING[player_position]
    player_ids_by_team = get_player_id_position(df, player_position_skillcorner)
    home_team_id = df["home_team_id"].iloc[0]
    away_team_id = df["away_team_id"].iloc[0]
    home_player_id = player_ids_by_team[home_team_id]
    away_player_id = player_ids_by_team[away_team_id]

    # ---- Build tracking datasets ----
    home_tr, home_players = build_team_tracking(df, "home")
    away_tr, away_players = build_team_tracking(df, "away")

    # ---- Calculate velocities ----
    home_tr = calculate_player_velocities(home_tr, home_players)
    away_tr = calculate_player_velocities(away_tr, away_players)

    # ---- Iterate over frames ----
    frames = list(df["frame"].unique())
    if sample_size:
        frames = random.sample(frames, sample_size)
    #frames_test = frames[:5]  # For rapid testing
    for frame in tqdm(frames, desc=f"Match {match_id}", leave=False):

        # Identify team in possession
        in_possession_team = df.loc[df["frame"] == frame, "team_in_possession"].values[0]

        # Determine concerned player_id
        if in_possession_team == home_team_id:
            player_id = home_player_id
            if player_id is None:
                continue  # No match -> skip
        elif in_possession_team == away_team_id:
            player_id = away_player_id
            if player_id is None:
                continue
        else:
            continue  # Invalid frame or no clear possession

        # ---- Full frame calculation ----
        player_frame_info = extract_player_pitch_control_and_contextual(
            player_id=player_id,
            frame=frame,
            home_tracking=home_tr,
            away_tracking=away_tr,
            home_players=home_players,
            away_players=away_players,
            params=params,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            resolution=resolution,
            half_pitch=True
        )

        # ---- Add global info ----
        player_frame_info.update({
            "frame": frame,
            "player_id": player_id,
            "player_position_role": player_position
        })

        all_results.append(player_frame_info)

    print(f"\n Processing complete: {len(all_results)} frames analyzed in total.")
    return all_results

def plot_half_pitch_individual_pitch_control(
    df,
    frame_results,
    sigma=1.0,
    pitch_length=PITCH_LENGTH,
    pitch_width=PITCH_WIDTH,
    half_pitch=True,
):
    """
    Plot Individual Pitch Control on a half pitch with contextual information.

    Args:
        df (pd.DataFrame): Match dataframe.
        frame_results (dict): Dictionary containing results for a specific frame.
        sigma (float): Smoothing factor for the gaussian filter.
        pitch_length (float): Length of the pitch.
        pitch_width (float): Width of the pitch.
        half_pitch (bool): Whether to plot half pitch or full pitch.
    """

    # ------------------------------------------------------------------
    # Frame filtering
    # ------------------------------------------------------------------
    frame_id = frame_results["frame"]
    df_frame = df[df["frame"] == frame_id]

    if df_frame.empty:
        raise ValueError(f"Frame {frame_id} not found in dataframe")

    # ------------------------------------------------------------------
    # Build tracking (Metrica-like)
    # ------------------------------------------------------------------
    home_tr, home_players = build_team_tracking(df, "home")
    away_tr, away_players = build_team_tracking(df, "away")

    home_frame = home_tr.loc[home_tr["frame"] == frame_id]
    away_frame = away_tr.loc[away_tr["frame"] == frame_id]

    # ------------------------------------------------------------------
    # Team names
    # ------------------------------------------------------------------
    home_team_id = df[f'home_team_id'].iloc[0]
    away_team_id = df[f'away_team_id'].iloc[0]
    
    home_team_name = (
        df_frame[df_frame["team_id"] == home_team_id]["team_short"].iloc[0]
    )
    away_team_name = (
        df_frame[df_frame["team_id"] == away_team_id]["team_short"].iloc[0]
    )

    # ------------------------------------------------------------------
    # Player info
    # ------------------------------------------------------------------
    target_player_id = frame_results["player_id"]
    player_row = df_frame[df_frame["player_id"] == target_player_id].iloc[0]

    player_name = player_row["player_short_name"]
    player_team_id = player_row["team_id"]
    player_color = "red" if player_team_id == home_team_id else "blue"

    # ------------------------------------------------------------------
    # Time (minute of game)
    # ------------------------------------------------------------------
    time_s = player_row["time_s"]
    minute = int(time_s // 60)

    # ------------------------------------------------------------------
    # Pitch Control (smoothed)
    # ------------------------------------------------------------------
    probs = np.array(frame_results["pitch_control_map"])
    probs_smooth = gaussian_filter(probs, sigma=sigma)

    # ------------------------------------------------------------------
    # Pitch
    # ------------------------------------------------------------------
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="white",
        line_color="black"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    if half_pitch:
        ax.set_xlim(pitch_length / 2, pitch_length)
        ax.set_ylim(0, pitch_width)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    x_edges = np.linspace(
        pitch_length / 2 if half_pitch else 0,
        pitch_length,
        probs.shape[1] + 1
    )
    y_edges = np.linspace(0, pitch_width, probs.shape[0] + 1)

    bin_statistic = dict(
        statistic=probs_smooth,
        x_grid=x_edges,
        y_grid=y_edges
    )

    pcm = pitch.heatmap(
        bin_statistic,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.9
    )

    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Individual Pitch Control Probability", fontsize=12)

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------
    def plot_players(frame_df, players, color, label):
        xs, ys = [], []
        for pid in players:
            x_col = f"x_rescaled_{pid}"
            y_col = f"y_rescaled_{pid}"
            if x_col in frame_df.columns:
                x, y = frame_df[x_col].values[0], frame_df[y_col].values[0]
                if not np.isnan(x):
                    xs.append(x)
                    ys.append(y)

        pitch.scatter(
            xs, ys,
            c=color,
            s=70,
            ax=ax,
            edgecolors="black",
            linewidth=0.6,
            label=label,
            zorder=5
        )

    plot_players(home_frame, home_players, "red", home_team_name)
    plot_players(away_frame, away_players, "blue", away_team_name)

    # ------------------------------------------------------------------
    # Target player (star)
    # ------------------------------------------------------------------
    x_star = player_row["x_rescaled"]
    y_star = player_row["y_rescaled"]

    pitch.scatter(
        x_star,
        y_star,
        marker="*",
        s=260,
        c=player_color,
        ax=ax,
        edgecolors="black",
        linewidth=1.5,
        zorder=7,
        label=player_name
    )

    # ------------------------------------------------------------------
    # Ball
    # ------------------------------------------------------------------
    ball = df_frame[df_frame["is_ball"]]
    if not ball.empty:
        pitch.scatter(
            ball["x_rescaled"].values[0],
            ball["y_rescaled"].values[0],
            c="black",
            s=60,
            ax=ax,
            zorder=8,
            label="Ball"
        )

    # ------------------------------------------------------------------
    # Defensive lines (filtered to half pitch)
    # ------------------------------------------------------------------
    x_min = pitch_length / 2 if half_pitch else 0
            
    defensive_lines = frame_results.get("defensive_lines", [])

    label_added = False
    for x_line in defensive_lines:
        if x_line >= x_min:
            ax.axvline(
                x=x_line,
                linestyle="--",
                color="grey",
                linewidth=2,
                alpha=0.7,
                label="Defensive lines" if not label_added else None
            )
            label_added = True

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        f"Individual Pitch Control (IPC) - {player_name} - {minute}'",
        fontsize=14
    )

    # ------------------------------------------------------------------
    # Infos text under plot
    # ------------------------------------------------------------------
    info_text = (
        f"Player in possession: {frame_results['in_possession']}\n"
        f"Distance to ball: {frame_results['distance_to_ball']:.2f} m\n"
        f"Nearest teammate: {frame_results['distance_to_nearest_teammate']:.2f} m\n"
        f"Nearest opponent: {frame_results['distance_to_nearest_opponent']:.2f} m"
    )

    ax.text(
        0.5,
        -0.02,
        info_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    ax.legend(loc="upper left")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    #plt.savefig(
    #    "images/ipc_example.png",
    #    dpi=300,               
    #    bbox_inches='tight',    
    #    facecolor='white'
    #)
    
    plt.show()

def process_match(
    file_path: str,
    player_position: str,
    game_situation: str,
    params: dict = PITCH_CONTROL_PARAMS,
    resolution: int = 1,
    save: bool = False,
    save_load_method: str = "gcp"
):
    """
    Function executed in parallel for a given match.
    
    Args:
        file_path (str): Path to the match file.
        player_position (str): Targeted player position.
        player_position_mapping (dict): Mapping for position names.
        game_situation (str): Specific game situation.
        params (dict): Simulation parameters.
        resolution (int): Grid resolution.
        save (bool): Whether to save the output.
        save_load_method (str): 'local' or 'gcp'.
    """
    match_id = Path(file_path).stem
    sub_dir = f"pitch_control_{player_position}_{game_situation[1]}"
    
    # Define paths based on method
    if save_load_method == "gcp":
        fs = gcsfs.GCSFileSystem()
        PITCH_CONTROL_DIR = f"{BASE_GCS_PATH}/{sub_dir}"
        outpath = f"{PITCH_CONTROL_DIR}/{match_id}.npz"
        
        if save and fs.exists(outpath):
            print(f" File already exists for match {match_id}")
            return match_id
            
    elif save_load_method == "local":
        PITCH_CONTROL_DIR = f"{BASE_LOCAL_PATH}/{sub_dir}"
        outpath = f"{PITCH_CONTROL_DIR}/{match_id}.npz"
        
        if save and os.path.exists(outpath):
            print(f" File already exists for match {match_id}")
            return match_id

    print(f" Processing match {match_id}...")
    try:
        results = extract_all_for_position_across_match(
            match_id=match_id,
            player_position=player_position,
            game_situation=game_situation,
            params=params,
            resolution=resolution,
            save_load_method=save_load_method
        )
        
        if save:
            if save_load_method == "gcp":
                save_npz_to_gcs({"results": results}, outpath)
                print(f" File saved: {outpath}")
            elif save_load_method == "local":
                os.makedirs(PITCH_CONTROL_DIR, exist_ok=True)
                np.savez(outpath, results=results)
                print(f" File saved: {outpath}")
                
    except Exception as e:
        print(f" Error on match {match_id} : {e}")
        
    return match_id

def process_all_matches_parallel(
    player_position: str,
    game_situation: tuple,
    pitch_control_model_params: dict = PITCH_CONTROL_PARAMS,
    pitch_control_resolution: int = 1,
    max_workers: int = 10,
    save: bool = False,
    save_load_method: str = "gcp"
):
    """
    Processes all detected matches in parallel to compute pitch control metrics.

    Args:
        player_position (str): The target player position to analyze.
        game_situation (tuple): A tuple (column, value) defining the specific game situation to filter.
        pitch_control_model_params (dict): Parameters for the pitch control model.
        pitch_control_resolution (int): The resolution grid size for pitch control calculations.
        max_workers (int): The maximum number of parallel workers.
        save (bool): Whether to save the output.
        save_load_method (str): 'local' or 'gcp'.
    """
    
    if save_load_method == "gcp":
        fs = gcsfs.GCSFileSystem()
        parquet_files = fs.glob(f"{PROCESSED_DIR}/*.parquet")
    elif save_load_method == "local":
        parquet_files = glob.glob(f"{PROCESSED_DIR_LOCAL}/*.parquet")

    print(f"{len(parquet_files)} matches detected")

    worker = partial(
        process_match,
        player_position=player_position,
        game_situation=game_situation,
        params=pitch_control_model_params,
        resolution=pitch_control_resolution,
        save=save,
        save_load_method=save_load_method
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        list(executor.map(worker, parquet_files))