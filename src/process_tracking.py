import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import requests
import traceback
from kloppy import skillcorner
import gcsfs

from .config import *
from .utils import write_parquet_gcs

def load_tracking_full_from_kloppy(
    match_id: int,
    sort_rows: bool = False,
    require_ball_detected: bool = True,
) -> pd.DataFrame:
    """
    Load full tracking data (players + ball) from SkillCorner via kloppy,
    rescale coordinates to pitch dimensions, and return a flat DataFrame.
    """

    dataset = skillcorner.load_open_data(
        match_id=match_id,
        coordinates="skillcorner",
    )

    rows = []

    for frame in dataset.frames:
        frame_id = frame.frame_id
        ts = frame.timestamp
        period = frame.period.id if frame.period is not None else None

        # --- Players ---
        for player, coords in frame.players_coordinates.items():
            is_detected = coords is not None

            rows.append({
                "match_id": match_id,
                "time": ts,
                "frame": frame_id,
                "period": period,
                "player_id": player.player_id,
                "is_detected": is_detected,
                "is_ball": False,
                "x": coords.x if is_detected else np.nan,
                "y": coords.y if is_detected else np.nan,
            })

        # --- Ball ---
        ball_coords = frame.ball_coordinates
        ball_detected = ball_coords is not None

        if (not require_ball_detected) or ball_detected:
            rows.append({
                "match_id": match_id,
                "time": ts,
                "frame": frame_id,
                "period": period,
                "player_id": -1,
                "is_detected": ball_detected,
                "is_ball": True,
                "x": ball_coords.x if ball_detected else np.nan,
                "y": ball_coords.y if ball_detected else np.nan,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # --- Type casting ---
    df["is_ball"] = df["is_ball"].astype(bool)
    df["is_detected"] = df["is_detected"].astype(bool)
    df["frame"] = df["frame"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    # --- Coordinate rescaling ---
    den_x, den_y = (X_MAX - X_MIN), (Y_MAX - Y_MIN)
    valid_xy = df[["x", "y"]].notna().all(axis=1)

    df["x_rescaled"] = np.where(
        valid_xy,
        (df["x"] - X_MIN) / den_x * PITCH_LENGTH,
        np.nan
    )
    df["y_rescaled"] = np.where(
        valid_xy,
        (df["y"] - Y_MIN) / den_y * PITCH_WIDTH,
        np.nan
    )

    # --- Sorting ---
    if sort_rows:
        df = df.sort_values(["frame", "is_ball", "player_id"]).reset_index(drop=True)
    else:
        df = df.sort_values(["frame"]).reset_index(drop=True)

    print(
        f"[match {match_id}] rows: {len(df)}, "
        f"players: {(~df['is_ball']).sum()}, "
        f"ball: {df['is_ball'].sum()}"
    )

    return df


def load_meta_from_github(match_id: int) -> pd.DataFrame:
    """
    Load SkillCorner match metadata from GitHub and format it to match
    the structure used by the legacy meta loader.
    """

    url = (
        "https://raw.githubusercontent.com/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_match.json"
    )

    r = requests.get(url)
    if r.status_code != 200:
        # File not found
        return pd.DataFrame()

    meta = r.json()
    if not meta:
        return pd.DataFrame()

    # Extract team information
    home_team = meta.get("home_team", {}) or {}
    away_team = meta.get("away_team", {}) or {}

    rows = []
    for p in meta.get("players", []):
        # Skip players without an id
        if "id" not in p:
            continue

        team_id = p.get("team_id")
        team = home_team if team_id == home_team.get("id") else away_team

        rows.append({
            "match_id": meta.get("id"),
            "player_id": p.get("id"),
            "team_id": team_id,
            "team_name": team.get("name"),
            "team_short": team.get("short_name"),
            "player_short_name": p.get("short_name"),
            "player_role": p.get("player_role", {}).get("name"),
            "position_group": p.get("player_role", {}).get("position_group"),
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
        })

    return pd.DataFrame(rows)


def load_events_basic_from_github(match_id: int) -> pd.DataFrame:
    """
    Load basic event data from SkillCorner GitHub and rescale coordinates
    to a 105x68 pitch.
    """

    url = (
        "https://raw.githubusercontent.com/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_dynamic_events.csv"
    )

    df = pd.read_csv(url)

    # Ensure required columns exist
    if "event_type" not in df.columns or "end_type" not in df.columns:
        return pd.DataFrame()

    # Example filtering logic kept commented for reference
    # df = df.query(
    #     "event_type == 'player_possession' and end_type == 'pass'"
    # ).copy()

    if df.empty:
        return pd.DataFrame()

    df["player_in_possession_id"] = df["player_id"]

    # Rescale SkillCorner coordinates to 105x68
    X_MIN, X_MAX = -52, 52
    Y_MIN, Y_MAX = -34, 34

    if {"x_start", "y_start", "x_end", "y_end"}.issubset(df.columns):
        df["x_start_rescaled"] = (df["x_start"] - X_MIN) / (X_MAX - X_MIN) * 105.0
        df["y_start_rescaled"] = (df["y_start"] - Y_MIN) / (Y_MAX - Y_MIN) * 68.0
        df["x_end_rescaled"] = (df["x_end"] - X_MIN) / (X_MAX - X_MIN) * 105.0
        df["y_end_rescaled"] = (df["y_end"] - Y_MIN) / (Y_MAX - Y_MIN) * 68.0

    df = df.loc[:, df.notna().any(axis=0)]

    if "frame_start" in df.columns:
        df = df.sort_values("frame_start").reset_index(drop=True)

    return df


def to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert time values to seconds.

    Supports:
    - strings like 'HH:MM:SS.ff'
    - numeric seconds
    - large numeric values interpreted as milliseconds
    """

    # Try numeric conversion first
    s_num = pd.to_numeric(series, errors="coerce")

    # Try timedelta parsing for formatted strings
    s_td = pd.to_timedelta(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )
    s_sec_td = s_td.dt.total_seconds()

    # Prefer timedelta values when available
    s = s_sec_td.fillna(s_num)

    # Detect millisecond-scale values and convert to seconds
    if s.notna().any():
        median_val = s.dropna().median()
        if median_val is not None and median_val > 1e6:
            s = s / 1000.0

    return s.astype(float)


def compute_all_velocities(
    tr_df: pd.DataFrame,
    fps_fallback: float = 10.0,
    interp_gap_s: float = 0.40,
    smooth_window_s: float = 0.70,
    clip_speed_player: float = 7.0,
    clip_speed_ball: float = 35.0,
) -> pd.DataFrame:
    """
    Compute velocity components and speed for all entities (players and ball),
    with interpolation, smoothing, and physical clipping.
    """

    df = tr_df.copy()
    df["time_s"] = to_seconds(df["time"])
    df = df.sort_values(["match_id", "is_ball", "player_id", "frame"]).reset_index(drop=True)

    results = []

    for (match_id, is_ball, pid), g in df.groupby(["match_id", "is_ball", "player_id"], sort=False):
        g = g.sort_values("frame").copy()
        if len(g) < 3:
            continue

        # Coordinate arrays
        x = pd.to_numeric(g["x_rescaled"], errors="coerce")
        y = pd.to_numeric(g["y_rescaled"], errors="coerce")
        t = pd.to_numeric(g["time_s"], errors="coerce")

        # Fallback when timestamps are missing
        if t.isna().any():
            dt_nominal = 1.0 / fps_fallback
            t = np.arange(len(x)) * dt_nominal

        dt = t.diff()
        dt_mean = np.nanmedian(dt)
        dt[dt <= 0] = dt_mean

        # Remove unrealistic temporal gaps
        mask_big_jump = dt > 3 * dt_mean
        if mask_big_jump.any():
            x[mask_big_jump] = np.nan
            y[mask_big_jump] = np.nan

        # Interpolation and smoothing
        x_i = x.interpolate(limit_direction="both", limit=3)
        y_i = y.interpolate(limit_direction="both", limit=3)

        window_frames = int(round(smooth_window_s / (dt_mean or (1.0 / fps_fallback))))
        if window_frames % 2 == 0:
            window_frames += 1

        x_s = x_i.rolling(window_frames, center=True, min_periods=1).median()
        y_s = y_i.rolling(window_frames, center=True, min_periods=1).median()

        # Velocity computation
        vx = x_s.diff() / dt
        vy = y_s.diff() / dt

        # Light velocity smoothing
        vx = vx.rolling(3, center=True, min_periods=1).mean()
        vy = vy.rolling(3, center=True, min_periods=1).mean()

        # Physical clipping
        vmax = clip_speed_ball if is_ball else clip_speed_player
        vx = vx.clip(lower=-vmax, upper=vmax)
        vy = vy.clip(lower=-vmax, upper=vmax)
        speed = np.sqrt(vx ** 2 + vy ** 2)

        g["vx_mps"] = vx
        g["vy_mps"] = vy
        g["speed_mps"] = speed
        results.append(g)

    df_vel = pd.concat(results, ignore_index=True)
    return df_vel


def merge_tracking_and_events(match_id: int, df_events: pd.DataFrame, df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tracking data with event information so that each pass start frame
    contains contextual labels (possession, teams, outcomes, coordinates).

    Only the start frame of each pass is used, as required for pre-pass modeling.
    """

    frames_all = []

    # Keep only possession-based pass events
    df_events_pass = df_events.query(
        "event_type == 'player_possession'"
    ).copy()

    if df_events_pass.empty:
        print(f" No valid pass events for match {match_id}")
        return pd.DataFrame()

    for _, ev in tqdm(df_events_pass.iterrows(), total=len(df_events_pass), desc=f"Merging match {match_id}"):
        frame_start = int(ev["frame_start"]) if "frame_start" in ev else None
        if frame_start is None or frame_start not in df_tracking["frame"].values:
            continue

        # Select tracking rows at the pass start frame
        tr_slice = df_tracking.loc[df_tracking["frame"] == frame_start].copy()
        if tr_slice.empty:
            continue

        # Broadcast event metadata to all entities in the frame
        tr_slice["event_id"] = ev.get("event_id")
        tr_slice["event_type"] = ev.get("event_type")
        tr_slice["end_type"] = ev.get("end_type")
        tr_slice["pass_outcome"] = ev.get("pass_outcome", None)
        tr_slice["team_in_possession_phase_type"] = ev.get("team_in_possession_phase_type", None)
        tr_slice["team_out_of_possession_phase_type"] = ev.get("team_out_of_possession_phase_type", None)
        tr_slice["player_in_possession_id"] = ev.get("player_in_possession_id", ev.get("player_id", None))
        tr_slice["team_in_possession"] = ev.get("team_id")
        tr_slice["team_in_possession_shortname"] = ev.get("team_shortname")
        tr_slice["attacking_side"] = ev.get("attacking_side")
        tr_slice["period"] = ev.get("period")

        # Pass start and end coordinates
        for c in ["x_start_rescaled", "y_start_rescaled", "x_end_rescaled", "y_end_rescaled"]:
            if c in ev:
                tr_slice[c] = ev[c]

        # Optional label columns for supervised learning
        for label_col in ["lead_to_goal", "lead_to_shot", "lead_to_box_entry"]:
            if label_col in ev:
                tr_slice[label_col] = ev[label_col]

        frames_all.append(tr_slice)

    if not frames_all:
        print(f" No matching frames for match {match_id}")
        return pd.DataFrame()

    df_merged = pd.concat(frames_all, ignore_index=True)

    # Final cleanup
    df_merged["match_id"] = match_id
    df_merged = df_merged.sort_values(["period", "frame"]).reset_index(drop=True)

    return df_merged


def normalize_tracking_direction(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Flip coordinates so that the team in possession always attacks from left to right.
    """

    df_norm = df_merged.copy()

    mask = df_norm["attacking_side"] == "right_to_left"
    df_norm.loc[mask, "x_rescaled"] = 105.0 - df_norm.loc[mask, "x_rescaled"]
    df_norm.loc[mask, "y_rescaled"] = 68.0 - df_norm.loc[mask, "y_rescaled"]
    df_norm.loc[mask, "vx_mps"] = -df_norm.loc[mask, "vx_mps"]
    df_norm.loc[mask, "vy_mps"] = -df_norm.loc[mask, "vy_mps"]

    return df_norm


def process_data(match_id: int, save_to_gcs=False) -> int:
    fs = gcsfs.GCSFileSystem()
    try:
        if save_to_gcs:
            outpath = f"{PROCESSED_DIR}/{match_id}.parquet"
            if fs.exists(outpath):
                print(f"{match_id} already processed")
                return match_id

        # 1) Load base data
        tr = load_tracking_full_from_kloppy(match_id)
        ev = load_events_basic_from_github(match_id)
        meta = load_meta_from_github(match_id)

        if tr is None or ev is None or meta is None or tr.empty or ev.empty or meta.empty:
            # Skip if any required input is missing
            return match_id

        # 2) Compute velocities
        tr = compute_all_velocities(tr)

        # 3) Merge metadata
        tr = tr.merge(meta, on=["match_id", "player_id"], how="left", validate="m:1")
        tr.loc[tr["is_ball"], ["team_id", "team_name", "team_short"]] = np.nan

        # 4) Merge with events
        df_merged = merge_tracking_and_events(match_id, ev, tr)

        # 5) Normalize attacking direction
        df_normed = normalize_tracking_direction(df_merged)

        # 6) Sanity checks and cleaning
        if df_normed[["x_rescaled", "y_rescaled"]].isna().all().any():
            raise ValueError("Missing rescaled coordinates after normalization.")

        df_normed = df_normed.dropna(subset=["x_rescaled", "y_rescaled"])

        # 7) Write to GCS
        if save_to_gcs:
            write_parquet_gcs(df_normed, outpath)

        return df_normed

    except Exception as e:
        # Log error but do not interrupt batch execution
        print(f" Error processing {match_id}: {e}")
        traceback.print_exc()
        return match_id
