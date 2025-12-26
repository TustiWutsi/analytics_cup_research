import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter

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
    percentiles of mean delta_weighted_xT within each dimension chosen.

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
    
    meta_out['game_situation'] = meta_out['game_situation'].replace({'defending_transition': 'transition'})
    
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
    ax.set_title(f"Percentile of iscT-Δ by player & {dimension_col}")

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