import fsspec
import gcsfs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset
import pytorch_lightning as pl
import tempfile
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from .config import *
from .utils import *


########## FUNCTIONS AND CLASSES TO CREATE SOCCERMAP CHANNELS TENSORS ##########

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------
def create_pitch_grid(dim=(68,105)):
    H, W = dim
    y = np.linspace(0, PITCH_WIDTH, H)
    x = np.linspace(0, PITCH_LENGTH, W)
    return np.meshgrid(x, y)   # shape (H,W)

# ------------------------------------------------------------
# 6 SPARSE CHANNELS: (att: loc,vx,vy) + (def: loc,vx,vy)
# ------------------------------------------------------------
def make_sparse_player_channels(df_frame, attacking_team, defending_team, dim=(68,105)):
    H, W = dim
    att = np.zeros((3, H, W), dtype=np.float32)
    deff = np.zeros((3, H, W), dtype=np.float32)

    for _, r in df_frame.iterrows():
        if r["is_ball"]: 
            continue

        x = int(np.clip(round(r["x_rescaled"] / PITCH_LENGTH * W), 0, W - 1))
        y = int(np.clip(round(r["y_rescaled"] / PITCH_WIDTH  * H), 0, H - 1))

        vx, vy = r.get("vx_mps", 0.0), r.get("vy_mps", 0.0)

        if r["team_short"] == attacking_team:
            att[0, y, x] = 1
            att[1, y, x] = vx
            att[2, y, x] = vy
        else:
            deff[0, y, x] = 1
            deff[1, y, x] = vx
            deff[2, y, x] = vy

    return att, deff

# ------------------------------------------------------------
# 2 DENSE CHANNELS: distances
# ------------------------------------------------------------
def make_distance_maps(ball_pos, dim=(68,105)):
    X, Y = create_pitch_grid(dim)
    dist_ball = np.sqrt((X - ball_pos[0])**2 + (Y - ball_pos[1])**2)
    dist_goal = np.sqrt((X - GOAL_CENTER[0])**2 + (Y - GOAL_CENTER[1])**2)
    return dist_ball.astype(np.float32), dist_goal.astype(np.float32)

# ------------------------------------------------------------
# 3 DENSE CHANNELS: angle maps (sin,cos,raw)
# ------------------------------------------------------------
def make_angle_maps(dim=(68,105)):
    H, W = dim
    X, Y = create_pitch_grid(dim)  # shape (H,W)

    # Vectors from each grid point to the top/bottom posts
    top_dx = GOAL_X - X
    top_dy = GOAL_Y_TOP - Y
    bot_dx = GOAL_X - X
    bot_dy = GOAL_Y_BOTTOM - Y

    top_angle = np.arctan2(top_dy, top_dx)
    bot_angle = np.arctan2(bot_dy, bot_dx)

    angle_open = np.abs(top_angle - bot_angle)
    angle_open = np.clip(angle_open, 0, np.pi)

    sin_map = np.sin(angle_open)
    cos_map = np.cos(angle_open)

    return sin_map.astype(np.float32), cos_map.astype(np.float32), angle_open.astype(np.float32)

# ------------------------------------------------------------
# 2 SPARSE CHANNELS: sin/cos(angle between ball-carrier
#                     vel and each teammate’s vel)
# ------------------------------------------------------------
def make_velocity_angle_channels(df_frame, ball_carrier_id, dim=(68,105)):
    H, W = dim
    sin_ch = np.zeros((H, W), dtype=np.float32)
    cos_ch = np.zeros((H, W), dtype=np.float32)

    carrier = df_frame[df_frame.player_id == ball_carrier_id]
    if carrier.empty:
        return sin_ch, cos_ch

    carrier = carrier.iloc[0]
    vx_c, vy_c = carrier.get("vx_mps", 0), carrier.get("vy_mps", 0)
    norm_c = np.hypot(vx_c, vy_c)

    if norm_c < 1e-6:
        return sin_ch, cos_ch

    for _, r in df_frame.iterrows():
        if r["is_ball"] or r["player_id"] == ball_carrier_id:
            continue

        x = int(np.clip(round(r["x_rescaled"] / PITCH_LENGTH * W), 0, W - 1))
        y = int(np.clip(round(r["y_rescaled"] / PITCH_WIDTH  * H), 0, H - 1))

        vx, vy = r.get("vx_mps", 0), r.get("vy_mps", 0)
        norm_t = np.hypot(vx, vy)
        if norm_t < 1e-6:
            continue

        dot = (vx*vx_c + vy*vy_c) / (norm_c * norm_t)
        dot = np.clip(dot, -1, 1)
        ang = np.arccos(dot)

        sin_ch[y, x] = np.sin(ang)
        cos_ch[y, x] = np.cos(ang)

    return sin_ch, cos_ch


# ------------------------------------------------------------
# FULL 13-CHANNEL MAP FOR ONE FRAME
# ------------------------------------------------------------
def generate_soccer_map(df_frame, attacking_team, defending_team, ball_pos, ball_carrier_id):
    att, deff = make_sparse_player_channels(df_frame, attacking_team, defending_team)
    dist_ball, dist_goal = make_distance_maps(ball_pos)
    sin_g, cos_g, ang_g = make_angle_maps()
    sin_v, cos_v = make_velocity_angle_channels(df_frame, ball_carrier_id)

    tensor = np.stack([
        *att, *deff,
        dist_ball, dist_goal,
        sin_g, cos_g, ang_g,
        sin_v, cos_v
    ], axis=0)  # (13, 68, 105)

    return tensor.astype(np.float32)

class ToSoccerMapPassSuccessTensorFromFrame:
    """
    Convert an entire frame (23 rows: 22 players + ball) into a 13-channel SoccerMap.
    Output: (13, H, W), mask (1, H, W), target scalar.
    """

    def __init__(self, dim=(68, 105)):
        self.H, self.W = dim   # grid resolution

    def _scale_coords(self, x, y):
        """Scale real pitch coords to grid indices"""
        xi = np.clip(np.round(x / PITCH_LENGTH * self.W), 0, self.W - 1).astype(int)
        yi = np.clip(np.round(y / PITCH_WIDTH  * self.H), 0, self.H - 1).astype(int)
        return xi, yi

    def __call__(self, df_frame):
        """
        df_frame = sous-dataframe des 23 rows d’un frame.
        """
        # ---- Extract infos identical for all rows ----
        start_x = df_frame["x_start_rescaled"].iloc[0]
        start_y = df_frame["y_start_rescaled"].iloc[0]
        end_x   = df_frame["x_end_rescaled"].iloc[0]
        end_y   = df_frame["y_end_rescaled"].iloc[0]

        target = float(df_frame["label"].iloc[0])
        ball_row = df_frame[df_frame["is_ball"]].iloc[0]
        ball_pos = np.array([ball_row["x_rescaled"], ball_row["y_rescaled"]])

        ball_carrier_id = df_frame["player_in_possession_id"].iloc[0]

        # Teams
        attacking = df_frame[df_frame["player_id"] == ball_carrier_id]["team_short"].iloc[0]
        defending = df_frame[df_frame["team_short"] != attacking]["team_short"].unique()[0]

        # ---- Build channels ----
        att_ch, def_ch = make_sparse_player_channels(df_frame, attacking, defending, dim=(self.H, self.W))
        dist_ball, dist_goal = make_distance_maps(ball_pos, dim=(self.H, self.W))
        sin_goal, cos_goal, angle_goal = make_angle_maps(dim=(self.H, self.W))
        sin_vel, cos_vel = make_velocity_angle_channels(df_frame, ball_carrier_id, dim=(self.H, self.W))

        tensor = np.stack([
            *att_ch,
            *def_ch,
            dist_ball, dist_goal,
            sin_goal, cos_goal, angle_goal,
            sin_vel, cos_vel
        ], axis=0)  # (13, H, W)

        # ---- mask: 1 at end location ----
        xe, ye = self._scale_coords(end_x, end_y)
        mask = np.zeros((1, self.H, self.W), dtype=np.float32)
        mask[0, ye, xe] = 1.0

        return (
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor([target], dtype=torch.float32),
        )
    
class SoccerMapPassSuccessDataset(Dataset):
    """
    Dataset où chaque item = un frame (23 rows).
    Expose :
      - self.df : dataframe source (copie)
      - self.groups : list of Index objects (one per frame)
      - self.frames : list of DataFrame per frame (convenience)
    """
    def __init__(self, df: pd.DataFrame, transform=None, dim=(68, 105)):
        self.df = df.copy()
        # check required columns (optionnel)
        required = ["match_id", "frame", "x_rescaled", "y_rescaled", "x_start_rescaled",
                    "y_start_rescaled", "x_end_rescaled", "y_end_rescaled", "is_ball"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")

        # grouper et garder uniquement frames avec 1 ballon
        grouped = self.df.groupby(["match_id", "frame"])
        self.groups = []
        self.frames = []
        for _, group in grouped:
            if group["is_ball"].sum() == 1:
                idx = group.index
                self.groups.append(idx)
                self.frames.append(group.reset_index(drop=True))  # sauvegarde une copy ré-indexée

        self.transform = transform if transform is not None else ToSoccerMapPassSuccessTensorFromFrame(dim=dim)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # on renvoie le tensor (x, mask, y) produit par la transform sur la sous-DF
        frame_df = self.frames[idx]  # déjà reset_index
        return self.transform(frame_df)

def build_soccer_map_dataloaders(
    fs,
    frames_dir: str,
    task: str,
    dataset_class,
    dim: tuple = (64, 50),
    batch_size: int = 64,
    balance_ratio: float | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Builds training and validation DataLoaders for xPass (pass probability) 
    or xThreat (goal probability) CNN training.

    This function handles:
    1. Loading and concatenating parquet files.
    2. Target label construction based on the specific task.
    3. Optional class balancing via random undersampling.
    4. Group-aware train/test splitting to prevent data leakage.

    Args:
        fs (fsspec.filesystem): The filesystem object (e.g., GCSFileSystem).
        frames_dir (str): Directory path containing *_normed.parquet files.
        task (str): The prediction task, either "pass" or "goal".
        dataset_class (class): The Dataset class to instantiate (e.g., SoccerMapPassSuccessDataset).
        dim (tuple): Input resolution for the CNN (height, width).
        batch_size (int): Batch size for the DataLoaders.
        balance_ratio (float | None): Ratio for undersampling the majority class. 
                                      If None, no balancing is performed.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed for random number generation.

    Returns:
        tuple: A tuple containing (train_loader, val_loader).
    """

    assert task in {"pass", "goal"}, "task must be 'pass' or 'goal'"

    # ---------------------------------------------------
    # Load & concatenate parquet files
    # ---------------------------------------------------
    all_frames = []
    files = fs.glob(f"{frames_dir}/*_normed.parquet")

    for fpath in files:
        with fs.open(fpath, "rb") as f:
            df = pd.read_parquet(f)
        all_frames.append(df)

    df = pd.concat(all_frames, ignore_index=True)

    # ---------------------------------------------------
    # Uniform label per (match_id, frame)
    # ---------------------------------------------------
    def aggregate_label(x):
        if (x == 1).any():
            return 1
        if (x == 0).any():
            return 0
        return np.nan

    df_label_agg = (
        df.groupby(["match_id", "frame"], as_index=False)["label"]
          .apply(aggregate_label)
          .rename(columns={"label": "label_uniform"})
    )

    df = df.merge(df_label_agg, on=["match_id", "frame"], how="left")
    df["label"] = df["label_uniform"]
    df = df.drop(columns=["label_uniform"]).drop_duplicates()

    # ---------------------------------------------------
    # Target construction
    # ---------------------------------------------------
    if task == "pass":
        df["label"] = df["pass_outcome"].map({
            "successful": 1.0,
            "unsuccessful": 0.0
        })

    elif task == "goal":
        df["label"] = df["lead_to_goal"].map({
            True: 1.0,
            False: 0.0
        })

    df = df[df["label"].isin([0.0, 1.0])]
    df = df.dropna(subset=["x_rescaled", "y_rescaled"])

    # ---------------------------------------------------
    # Optional balancing (frame-level)
    # ---------------------------------------------------
    # 
    if balance_ratio is not None:
        pos_df = df[df["label"] == 1.0]
        neg_df = df[df["label"] == 0.0]

        pos_frames = pos_df["frame"].unique()
        neg_frames = neg_df["frame"].unique()

        if len(pos_frames) <= len(neg_frames):
            minority_frames = pos_frames
            majority_frames = neg_frames
            minority_df = pos_df
            majority_df = neg_df
        else:
            minority_frames = neg_frames
            majority_frames = pos_frames
            minority_df = neg_df
            majority_df = pos_df

        n_target_majority = int(len(minority_frames) * balance_ratio)
        n_target_majority = min(n_target_majority, len(majority_frames))

        sampled_majority_frames = (
            pd.Series(majority_frames)
              .sample(n=n_target_majority, random_state=random_state)
              .values
        )

        majority_sampled_df = majority_df[
            majority_df["frame"].isin(sampled_majority_frames)
        ]

        df = pd.concat([minority_df, majority_sampled_df], ignore_index=True)

    # ---------------------------------------------------
    # Train / validation split by (match_id, frame)
    # ---------------------------------------------------
    # 
    groups = df[["match_id", "frame"]].apply(
        lambda r: f"{r['match_id']}__{r['frame']}", axis=1
    )

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # ---------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------
    train_dataset = dataset_class(train_df, dim=dim)
    val_dataset = dataset_class(val_df, dim=dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
    
########## SOCCERMAP MODELS CLASSES ##########
    
class SoccerMap(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(1, 32),  # remplace BatchNorm2d
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU()
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -1.0, 1.0)
        x = self.encoder(x)
        x = self.head(x)
        return x  # [B, 1, H, W]
    


class PytorchSoccerMapModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters() # saves lr=1e-4 automatically, so it’s logged and checkpointed.

        # --- Core SoccerMap CNN ---
        self.model = SoccerMap(in_channels=13)

        # --- Loss & Metrics ---
        self.criterion = nn.BCEWithLogitsLoss() # The main training objective — compares predictions vs. true labels (0 or 1). It’s a binary cross-entropy loss that expects raw outputs (logits) from the model.
        self.train_acc = torchmetrics.classification.BinaryAccuracy() # How often the model’s prediction is correct (above or below 0.5).
        self.train_auc = torchmetrics.classification.BinaryAUROC() # Measures how well the model ranks positives above negatives — more robust than accuracy for imbalanced data.
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        
    def on_train_start(self):
        print(f"🚀 Training on device: {self.device}")
        # Si tu veux logguer la mémoire GPU :
        if self.device.type == "cuda":
            print(f"🔋 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def forward(self, x):
        """Forward pass through CNN to get full surface logits. “To make a prediction, pass the input x through the CNN.”"""
        return self.model(x)  # [B, 1, 68, 105]

    def shared_step(self, batch, stage="train"):
        x, mask, y = batch
        logits = self(x)  # [B, 1, 68, 105] runs the CNN to produce the surface logits (raw values before sigmoid).

        # --- Apply mask to reduce spatial logits to one scalar per sample ---
        masked_logits = (logits * mask).sum(dim=(2, 3))  # [B, 1]
        """extracts only the predicted value at the pass destination cell.
        The rest of the pitch is ignored (set to zero).
        The sum effectively collapses the spatial surface into a single scalar prediction per sample.
        So now you’ve gone from [B, 1, 68, 105] → [B, 1]."""

        # --- Compute loss ---
        loss = self.criterion(masked_logits, y)

        # --- Compute metrics ---
        probs = torch.sigmoid(masked_logits) # sigmoid converts raw numbers into probabilities (0–1 range).
        acc = self.train_acc(probs, y.int()) if stage == "train" else self.val_acc(probs, y.int())
        auc = self.train_auc(probs, y.int()) if stage == "train" else self.val_auc(probs, y.int())

        # --- Logging ---
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/auc", auc, prog_bar=True, on_epoch=True)
        """Lightning handles logging automatically — these values will appear in your progress bar and epoch summaries."""

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")
    
    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train/loss_epoch")
        acc = self.trainer.callback_metrics.get("train/acc_epoch")
        auc = self.trainer.callback_metrics.get("train/auc_epoch")
        print(f"🧠 Epoch {self.current_epoch} — TRAIN | loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}")

    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get("val/loss")
        acc = self.trainer.callback_metrics.get("val/acc")
        auc = self.trainer.callback_metrics.get("val/auc")
        print(f"🎯 Epoch {self.current_epoch} — VAL | loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # The optimizer updates all model parameters based on gradients from the loss function.
    
########## FUNCTIONS TO RUN MODEL PREDICTIONS, CALCULATE NEW METRICS AND PLOT THEM ON PITCH MAPS ##########
    
def predict_maps(
    match_id: int,
    df: pd.DataFrame = None,
    frame_results: list = None,
    model_local: bool = True,
    # frame: int,
    sigma: float = 2.0,
):
    """
    Generates and plots 4 half-pitch visualization maps for a specific match frame:
    1) Individual Pitch Control (IPC)
    2) xPass (Expected Pass probability)
    3) xThreat (Expected Threat)
    4) IPC * xPass * xThreat

    Also calculates weighted xT metrics based on the interaction of these maps.

    Args:
        match_id (int): The ID of the match to analyze.
        tensorizer: Function to convert dataframe rows into model input tensors.
        sigma (float): Standard deviation for Gaussian smoothing of the maps.

    Returns:
        tuple:
            - iscT (float): The weighted Expected Threat scalar.
            - iscT_delta (float): The weighted xT relative to the ball's current xT.
    """

    # --------------------------------------------------
    #  Pitch control for this frame
    # --------------------------------------------------
    
    if frame_results is None:
        pitch_control_path = f"{PITCH_CONTROL_DIR}/{match_id}.npz"
        with fs.open(pitch_control_path, "rb") as f:
            pitch_control_results = np.load(f, allow_pickle=True)["results"].tolist()
    else:
        pitch_control_results = frame_results
    
    # pc_entry = next(
    #    d for d in pitch_control_results
    #    if d["frame"] == frame
    # )
    pc_entry = pitch_control_results[0]
    frame = pc_entry['frame']

    pitch_control = pc_entry["pitch_control_map"]  # (32,50)
    pitch_control_smooth = gaussian_filter(pitch_control, sigma=sigma)
    
    player_id = pc_entry["player_id"]

    # --------------------------------------------------
    #  Model predictions (FULL pitch)
    # --------------------------------------------------
    if df is None:
        df = read_parquet_gcs(fs, gcs_path=f"{PROCESSED_DIR}/{match_id}.parquet")
    frame_df = df.query("frame == @frame")
    frame_df['label'] = 0
    
    tensorizer = ToSoccerMapPassSuccessTensorFromFrame(dim=(64, 50))
    x, mask, _ = tensorizer(frame_df)
    x = x.unsqueeze(0)
    
    if model_local:
        best_ckpt_xpass_path = f"{MODELS_DIR}/best_xpass.ckpt"
        best_ckpt_xthreat_path = f"{MODELS_DIR}/best_xthreat.ckpt"
        
        model_xpass = load_checkpoint_from_local(best_ckpt_xpass_path, PytorchSoccerMapModel)
        model_xthreat = load_checkpoint_from_local(best_ckpt_xthreat_path, PytorchSoccerMapModel)
    else:
        best_ckpt_xpass_path = f"{SOCCERMAP_MODELS_DIR}/best_xpass.ckpt"
        best_ckpt_xthreat_path = f"{SOCCERMAP_MODELS_DIR}/best_xthreat.ckpt"

        model_xpass = load_checkpoint_from_gcs(best_ckpt_xpass_path, PytorchSoccerMapModel)
        model_xthreat = load_checkpoint_from_gcs(best_ckpt_xthreat_path, PytorchSoccerMapModel)

    model_xpass.eval()
    model_xthreat.eval()

    with torch.no_grad():
        xpass = torch.sigmoid(model_xpass(x))[0, 0].cpu().numpy()    # (64,50)
        xthreat = torch.sigmoid(model_xthreat(x))[0, 0].cpu().numpy()

    # --------------------------------------------------
    #  Keep ONLY right half of pitch
    # --------------------------------------------------
    H = xpass.shape[0]  # 64
    xpass_half = xpass[H//2:, :]      # (32,50)
    xpass_smooth = gaussian_filter(xpass, sigma=sigma)
    
    xthreat_half = xthreat[H//2:, :]  # (32,50)
    xthreat_smooth = gaussian_filter(xthreat, sigma=sigma)

    # --------------------------------------------------
    #  iscT computation
    # --------------------------------------------------
    # 
    isct = pitch_control * xpass_half * xthreat_half
    isct_smooth = gaussian_filter(isct, sigma=sigma)

    # --------------------------------------------------
    #  iscT
    # --------------------------------------------------
    weight = pitch_control * xpass_half
    iscT = np.sum(weight * xthreat_half) / (np.sum(weight) + 1e-8)
    
    # ==================================================
    #  iscT_delta
    # ==================================================
    # Convert ball position to grid index
    # 
    ball = frame_df[frame_df["is_ball"]].iloc[0]
    bx = int(np.clip(round((ball["x_rescaled"] - 105 / 2) / (105 / 2) * 50), 0, 49))
    by = int(np.clip(round(ball["y_rescaled"] / 68 * 32), 0, 31))

    xT_ball = xthreat_half[by, bx]

    delta_xT = xthreat_half - xT_ball
    iscT_delta = np.sum(weight * delta_xT) / (np.sum(weight) + 1e-8)

    # --------------------------------------------------
    # Identify attacking and defending players
    # --------------------------------------------------
    team_att = frame_df["team_in_possession"].iloc[0]

    attackers = frame_df[(frame_df["team_id"] == team_att) & (~frame_df["is_ball"])]
    defenders = frame_df[(frame_df["team_id"] != team_att) & (~frame_df["is_ball"])]
    ball = frame_df[frame_df["is_ball"]]

    player_row = frame_df[frame_df["player_id"] == player_id].iloc[0]

    player_name = player_row.get("player_short_name", f"Player {player_id}")

    team_names = frame_df.loc[~frame_df["is_ball"], "team_name"].unique()
    match_name = " vs ".join(team_names) if len(team_names) == 2 else f"Match {match_id}"

    # --------------------------------------------------
    #  Plot
    # --------------------------------------------------
    
    pitch_length=105
    pitch_width=68
    
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="white",
        line_color="black"
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    titles = [
        "Individual Pitch Control (IPC)",
        "xPass",
        "xT",
        "IPC × xPass × xThreat"
    ]

    maps = [
        pitch_control,
        xpass,
        xthreat,
        isct
    ]
    
    maps_smoothed = [
        pitch_control_smooth,
        xpass_smooth,
        xthreat_smooth,
        isct_smooth
    ]
    
    norms = [
        None,
        colors.Normalize(vmin=0.5, vmax=0.95, clip=True),
        colors.Normalize(vmin=0.01, vmax=0.1, clip=True),
        None
    ]

    for ax, mat, title, norm in zip(axes, maps_smoothed, titles, norms):
        pitch.draw(ax=ax)

        ax.set_xlim(pitch_length/2, pitch_length)
        ax.set_ylim(0, pitch_width)

        im = ax.imshow(
            mat,
            extent=[pitch_length/2, pitch_length, 0, pitch_width],
            origin="lower",
            cmap="viridis",
            norm=norm,
            alpha=0.85,
            zorder=1
        )

        # Players
        ax.scatter(
            attackers["x_rescaled"], attackers["y_rescaled"],
            c="red", s=80, edgecolors="black", zorder=3, label="Attacking"
        )
        ax.scatter(
            defenders["x_rescaled"], defenders["y_rescaled"],
            c="blue", s=80, edgecolors="black", zorder=3, label="Defending"
        )

        # Ball
        ax.scatter(ball["x_rescaled"], ball["y_rescaled"],
                   c="white", s=120, edgecolors="black", zorder=4)

        # Target player 
        ax.scatter(player_row["x_rescaled"], player_row["y_rescaled"],
                   marker="*", s=220, c="red", edgecolors="black", zorder=5)

        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.035)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.suptitle(
        f"Indivisual Space Controlled Threat (iscT) maps decomposition — {match_name}\nFocus Player: {player_name}",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()

    return iscT, iscT_delta

def compute_metrics(
    meta: pd.DataFrame,
    df: pd.DataFrame = None,
    frame_results: list = None,
    model_local: bool = True,
    batch_size: int = 1,
    save_to_gcs: bool = False
):
    """
    Computes `iscT` and `iscT_delta` for an entire dataset using batch processing.
    
    Iterates through metadata, groups by match ID to minimize I/O operations, loads 
    pitch control and tracking data once per match, and runs model inference in batches.

    Args:
        meta (pd.DataFrame): Metadata dataframe containing match_ids and frames to process.
        batch_size (int): Number of frames to process in a single inference pass.

    Returns:
        pd.DataFrame: The updated metadata dataframe with computed metrics columns.
    """
    
    if model_local:
        best_ckpt_xpass_path = f"{MODELS_DIR}/best_xpass.ckpt"
        best_ckpt_xthreat_path = f"{MODELS_DIR}/best_xthreat.ckpt"
        
        model_xpass = load_checkpoint_from_local(best_ckpt_xpass_path, PytorchSoccerMapModel)
        model_xthreat = load_checkpoint_from_local(best_ckpt_xthreat_path, PytorchSoccerMapModel)
    else:
        best_ckpt_xpass_path = f"{SOCCERMAP_MODELS_DIR}/best_xpass.ckpt"
        best_ckpt_xthreat_path = f"{SOCCERMAP_MODELS_DIR}/best_xthreat.ckpt"

        model_xpass = load_checkpoint_from_gcs(best_ckpt_xpass_path, PytorchSoccerMapModel)
        model_xthreat = load_checkpoint_from_gcs(best_ckpt_xthreat_path, PytorchSoccerMapModel)
    
    model_xpass.eval()
    model_xthreat.eval()

    meta_out = meta.copy()
    meta_out["iscT"] = np.nan
    meta_out["iscT_delta"] = np.nan
    
    player_position = meta_out.player_position_role.iloc[0].lower().replace(" ", "_")
    game_situation = meta_out.game_situation.iloc[0].lower().replace(" ", "_")

    # --------------------------------------------------
    # Group by match_id (CRUCIAL for performance)
    # --------------------------------------------------
    for match_id, meta_match in tqdm(meta.groupby("match_id"), desc="Matches"):

        # ---------- Pitch control ----------
        if frame_results is None:
            PITCH_CONTROL_DIR = f"{BASE_GCS_PATH}/pitch_control_{player_position}_{game_situation}"
            pitch_control_path = f"{PITCH_CONTROL_DIR}/{match_id}.npz"
            if not fs.exists(pitch_control_path):
                continue
    
            with fs.open(pitch_control_path, "rb") as f:
                pc_results = np.load(f, allow_pickle=True)["results"].tolist()
        else:
            pc_results = frame_results

        pc_by_frame = {d["frame"]: d for d in pc_results}

        # ---------- Frame data ----------
        if df is None:
            df_path = f"{PROCESSED_DIR}/{match_id}.parquet"
            if not fs.exists(df_path):
                continue
    
            df_match = read_parquet_gcs(fs, df_path)
        else:
            df_match = df

        # ---------- Loop frames (batched) ----------
        tensors = []
        frame_refs = []

        for idx, row in meta_match.iterrows():
            frame = row["frame"]

            if frame not in pc_by_frame:
                continue

            frame_df = df_match[df_match["frame"] == frame]
            if frame_df.empty or frame_df["is_ball"].sum() != 1:
                continue

            frame_df = frame_df.copy()
            frame_df["label"] = 0
            
            tensorizer = ToSoccerMapPassSuccessTensorFromFrame(dim=(64, 50))
            x, _, _ = tensorizer(frame_df)
            tensors.append(x)
            frame_refs.append((idx, frame_df, pc_by_frame[frame]))

            # ---------- Run batch ----------
            if len(tensors) == batch_size:
                _run_batch(
                    tensors,
                    frame_refs,
                    model_xpass,
                    model_xthreat,
                    meta_out,
                )
                tensors, frame_refs = [], []

        # ---------- Last batch ----------
        if tensors:
            _run_batch(
                tensors,
                frame_refs,
                model_xpass,
                model_xthreat,
                meta_out,
            )
    
    if save_to_gcs:
        RESULTS_DIR = f"{BASE_GCS_PATH}/results"
        output_path = f"{RESULTS_DIR}/results_{player_position}_{game_situation}.parquet"
        write_parquet_gcs(meta_out, output_path)

    return meta_out

def _run_batch(
    tensors,
    frame_refs,
    model_xpass,
    model_xthreat,
    meta_out,
):
    """
    Internal helper to execute model inference on a batch and update the results DataFrame.

    Args:
        tensors (list): List of input tensors for the models.
        frame_refs (list): List of tuples (index, frame_df, pc_entry) corresponding to the tensors.
        model_xpass: The xPass model.
        model_xthreat: The xThreat model.
        meta_out (pd.DataFrame): The dataframe to update in-place.
    """
    X = torch.stack(tensors)  # (B, C, 64, 50)

    with torch.no_grad():
        xpass_full = torch.sigmoid(model_xpass(X))[:, 0].cpu().numpy()
        xthreat_full = torch.sigmoid(model_xthreat(X))[:, 0].cpu().numpy()

    for i, (meta_idx, frame_df, pc_entry) in enumerate(frame_refs):

        pitch_control = pc_entry["pitch_control_map"]  # (32,50)

        # Right half
        xpass_half = xpass_full[i, 32:, :]
        xthreat_half = xthreat_full[i, 32:, :]

        # Ball xT
        ball = frame_df[frame_df["is_ball"]].iloc[0]
        xb = int(np.clip(ball["x_rescaled"] / 105 * 64, 0, 63))
        yb = int(np.clip(ball["y_rescaled"] / 68 * 50, 0, 49))
        xT_ball = xthreat_full[i, xb, yb]

        # Metrics
        weight = pitch_control * xpass_half
        denom = np.sum(weight) + 1e-8

        meta_out.at[meta_idx, "iscT"] = np.sum(weight * xthreat_half) / denom
        meta_out.at[meta_idx, "iscT_delta"] = (
            np.sum(weight * (xthreat_half - xT_ball)) / denom
        )