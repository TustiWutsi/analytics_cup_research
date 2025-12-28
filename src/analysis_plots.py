import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter

from .config import *
from .utils import *

def plot_isct_delta_by_player_single_match(df):
    match_id = df["match_id"].iloc[0]
    player_position = df["player_position_role"].iloc[0]
    game_situation = df["game_situation"].iloc[0]

    players = df["player_name"].unique()
    n_players = len(players)

    n_cols = min(3, n_players)
    n_rows = int(np.ceil(n_players / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for ax, player in zip(axes.flatten(), players):
        values = df.loc[df["player_name"] == player, "iscT_delta"]

        mean = values.mean()
        std = values.std()

        ax.hist(values, bins=30, density=True, alpha=0.7)

        ax.axvline(mean, linestyle="--", linewidth=2, label=f"μ = {mean:.2f}")

        ax.axvline(mean - std, linestyle=":", linewidth=1, label=f"σ = {std:.2f}")
        ax.axvline(mean + std, linestyle=":", linewidth=1)

        ax.set_title(player)
        ax.set_xlabel("iscT_delta")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for ax in axes.flatten()[len(players):]:
        ax.axis("off")

    fig.suptitle(
        f"iscT_delta of {player_position} when {game_situation} during the game {match_id}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_player_percentiles(
    meta_out: pd.DataFrame,
    value_col: str = "iscT_delta",
    dimension_col : str = "game_situation",
    game_situation_filter: str = None,
    cmap: str = "viridis",
    figsize=(14, 10),
):
    """
    Plot a heatmap (players × game_situations) where values are
    percentiles of mean iscT_delta within each dimension chosen.

    Parameters
    ----------
    meta_out : pd.DataFrame
        Must contain:
        - player_name
        - game_situation
        - iscT_delta (or chosen value_col)

    value_col : str
        Column on which percentiles are computed

    cmap : str
        Colormap for heatmap

    Returns
    -------
    pivot_percentiles : pd.DataFrame
        Table of percentiles (players x game_situations)
    """
    
    meta_out['game_situation'] = meta_out['game_situation'].replace({
        'defending_transition': 'transition',
        'defending_quick_break': 'quick_break'
    })
    
    if game_situation_filter:
        meta_out = meta_out[meta_out.game_situation == game_situation_filter]

    # ---------------------------------------------------
    # 1) Mean delta per (player, dimension chosen)
    # ---------------------------------------------------
    mean_table = (
        meta_out
        .groupby(["player_name", dimension_col], as_index=False)[value_col]
        .mean()
    )

    # ---------------------------------------------------
    # 2) Convert to percentiles
    # ---------------------------------------------------
    mean_table["percentile"] = (
        mean_table
        .groupby(dimension_col)[value_col]
        .rank(pct=True)
        * 100
    )

    # ---------------------------------------------------
    # 3) Pivot to matrix form
    # ---------------------------------------------------
    pivot_percentiles = mean_table.pivot(
        index="player_name",
        columns=dimension_col,
        values="percentile"
    )

    # ---------------------------------------------------
    # 4) Plot heatmap
    # ---------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    norm = colors.Normalize(vmin=0, vmax=100)

    im = ax.imshow(
        pivot_percentiles.values,
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    # Axis ticks
    ax.set_xticks(np.arange(pivot_percentiles.shape[1]))
    ax.set_yticks(np.arange(pivot_percentiles.shape[0]))

    ax.set_xticklabels(pivot_percentiles.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot_percentiles.index)

    ax.set_xlabel(dimension_col)
    ax.set_ylabel("Player")
    if dimension_col == 'cluster_gathered':
        ax.set_title(f"Percentile of iscT-Δ by cluster - Attacking against {game_situation_filter} defense")
    else:
        ax.set_title(f"Percentile of iscT-Δ by {dimension_col}")

    # ---------------------------------------------------
    # 5) Annotate each cell with percentile value
    # ---------------------------------------------------
    for i in range(pivot_percentiles.shape[0]):
        for j in range(pivot_percentiles.shape[1]):
            val = pivot_percentiles.iloc[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035)
    cbar.set_label(f"Percentile (within {dimension_col})", fontsize=11)

    plt.tight_layout()
    plt.show()

    return pivot_percentiles
    
def plot_top_bottom_isct_delta(
    player_name: str,
    player_position: str,
    game_situation: tuple,
    sigma: float = 2.0,
    save_load_method: str = "gcp"
):
    """
    Plot 6 half-pitches:
    - top row: 3 frames with highest iscT_delta
    - bottom row: 3 frames with lowest iscT_delta

    Heatmap = pitch control (smoothed)
    """

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    if save_load_method == "gcp":
        base_path = BASE_GCS_PATH
        fs = gcsfs.GCSFileSystem()
    else:
        base_path = BASE_LOCAL_PATH
        fs = None

    RESULTS_DIR = f"{base_path}/results"
    PROCESSED_DIR = f"{base_path}/processed"
    PITCH_CONTROL_DIR = f"{base_path}/pitch_control_{player_position}_{game_situation[1]}"

    # -------------------------------------------------
    # Load meta
    # -------------------------------------------------
    meta_path = f"{RESULTS_DIR}/results_{player_position}_{game_situation[1]}.parquet"

    if save_load_method == "gcp":
        with fs.open(meta_path, "rb") as f:
            meta = pd.read_parquet(f)
    else:
        meta = pd.read_parquet(meta_path)

    meta_sub = meta[
        (meta["player_name"] == player_name) &
        (meta["game_situation"] == game_situation[1])
    ].copy()

    # remove NaNs and in-possession frames
    meta_sub = meta_sub.dropna(subset=["iscT_delta"])
    #meta_sub = meta_sub[meta_sub["in_possession"] == False]

    if len(meta_sub) == 0:
        raise ValueError("No valid frames after filtering.")

    # -------------------------------------------------
    # Select TOP / BOTTOM 3
    # -------------------------------------------------
    top3 = meta_sub.sort_values("iscT_delta", ascending=False).head(3)
    bot3 = meta_sub.sort_values("iscT_delta", ascending=True).head(3)

    selected = pd.concat([top3, bot3], axis=0).reset_index(drop=True)

    # -------------------------------------------------
    # Pitch
    # -------------------------------------------------
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="white",
        line_color="black"
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    pcm_for_cbar = None

    # -------------------------------------------------
    # Loop over frames
    # -------------------------------------------------
    for i, row in selected.iterrows():
        ax = axes[i]

        match_id = row["match_id"]
        frame = row["frame"]
        delta_xt = row["iscT_delta"]

        # ---------------------------
        # Load pitch control
        # ---------------------------
        pc_path = f"{PITCH_CONTROL_DIR}/{match_id}.npz"

        try:
            if save_load_method == "gcp":
                with fs.open(pc_path, "rb") as f:
                    data = np.load(f, allow_pickle=True)
                    pc_results = data["results"].tolist()
            else:
                with open(pc_path, "rb") as f:
                    data = np.load(f, allow_pickle=True)
                    pc_results = data["results"].tolist()
        except Exception:
            continue

        pc_entry = next((d for d in pc_results if d["frame"] == frame), None)
        if pc_entry is None:
            continue

        ## Skip if player had possession
        #if pc_entry.get("in_possession", False):
        #    continue

        pitch_control = np.array(pc_entry["pitch_control_map"])  # (32, 50)
        pitch_control_smooth = gaussian_filter(pitch_control, sigma=sigma)

        player_id = pc_entry["player_id"]

        # ---------------------------
        # Load frame dataframe
        # ---------------------------
        df_path = f"{PROCESSED_DIR}/{match_id}.parquet"

        if save_load_method == "gcp":
            with fs.open(df_path, "rb") as f:
                df = pd.read_parquet(f)
        else:
            df = pd.read_parquet(df_path)

        frame_df = df[df["frame"] == frame]

        team_in_pos = frame_df["team_in_possession"].iloc[0]

        attackers = frame_df[
            (frame_df["team_id"] == team_in_pos) & (~frame_df["is_ball"])
        ]
        defenders = frame_df[
            (frame_df["team_id"] != team_in_pos) & (~frame_df["is_ball"])
        ]

        ball = frame_df[frame_df["is_ball"]]
        player_row = frame_df[frame_df["player_id"] == player_id]

        # ---------------------------
        # Draw pitch
        # ---------------------------
        pitch.draw(ax=ax)
        ax.set_xlim(105 / 2, 105)
        ax.set_ylim(0, 68)

        # ---------------------------
        # Heatmap (correct mplsoccer usage)
        # ---------------------------
        x_edges = np.linspace(105 / 2, 105, pitch_control_smooth.shape[1] + 1)
        y_edges = np.linspace(0, 68, pitch_control_smooth.shape[0] + 1)

        bin_statistic = dict(
            statistic=pitch_control_smooth,
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
        
        pcm_for_cbar = pcm

        # ---------------------------
        # Players
        # ---------------------------
        ax.scatter(
            attackers["x_rescaled"], attackers["y_rescaled"],
            c="red", s=60, edgecolors="black", zorder=3, label="Attacking"
        )

        ax.scatter(
            defenders["x_rescaled"], defenders["y_rescaled"],
            c="blue", s=60, edgecolors="black", zorder=3, label="Defending"
        )

        # Ball
        if not ball.empty:
            ax.scatter(
                ball["x_rescaled"], ball["y_rescaled"],
                c="white", s=30, edgecolors="black", zorder=4
            )

        # Target player (star, same color as team)
        if not player_row.empty:
            ax.scatter(
                player_row["x_rescaled"],
                player_row["y_rescaled"],
                marker="*",
                s=260,
                c="red",
                edgecolors="black",
                zorder=5
            )

        # Velocity arrows (small)
        ax.quiver(
            attackers["x_rescaled"], attackers["y_rescaled"],
            attackers["vx_mps"], attackers["vy_mps"],
            color="darkred",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.008,
            zorder=4
        )

        ax.quiver(
            defenders["x_rescaled"], defenders["y_rescaled"],
            defenders["vx_mps"], defenders["vy_mps"],
            color="navy",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.008,
            zorder=4
        )

        ax.set_title(f"iscT-Δ = {delta_xt:.4f}", fontsize=11)

    # -------------------------------------------------
    # Row labels
    # -------------------------------------------------
    fig.text(0.01, 0.73, "TOP 3", fontsize=14, weight="bold", rotation=90)
    fig.text(0.01, 0.28, "BOTTOM 3", fontsize=14, weight="bold", rotation=90)

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Attacking team",
               markerfacecolor="red", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Defending team",
               markerfacecolor="blue", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="*", color="w", label="Target player",
               markerfacecolor="red", markeredgecolor="black", markersize=14),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        f"{player_name} — Attacking against {game_situation[1]} - Top3/Bottom3 iscT-Δ",
        fontsize=16,
        y=0.98
    )

    # -------------------------------------------------
    # Adjustment & Colorbar
    # -------------------------------------------------
    plt.tight_layout()
    
    plt.subplots_adjust(right=0.88, bottom=0.1)

    if pcm_for_cbar is not None:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7]) 
        
        cbar = fig.colorbar(
            pcm_for_cbar,
            cax=cbar_ax
        )
        cbar.set_label("Individual Pitch Control Probability", fontsize=12)
        
    plt.savefig(
        "images/isct_delta_top3_bottom3.png",               # Le chemin/nom du fichier
        dpi=300,                # Haute résolution (300 est standard pour l'impression/web propre)
        bbox_inches='tight',    # CRUCIAL : empêche que tes légendes ou titres soient coupés
        facecolor='white'       # Force le fond blanc (parfois transparent par défaut)
    )

    plt.show()