import pandas as pd
import gcsfs
import io
import pytorch_lightning as pl

fs = gcsfs.GCSFileSystem(token="google_default")

def read_parquet_gcs(fs: gcsfs.GCSFileSystem, gcs_path: str) -> pd.DataFrame:
    if not fs.exists(gcs_path):
        raise FileNotFoundError(gcs_path)
    with fs.open(gcs_path, "rb") as f:
        return pd.read_parquet(f)
    
    
def write_parquet_gcs(df: pd.DataFrame, gcs_path: str):
    df.to_parquet(gcs_path, index=False)
    

def save_npz_to_gcs(array_dict: dict, gcs_path: str):
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **array_dict)
        buffer.seek(0)
        with fsspec.open(gcs_path, "wb") as f:
            f.write(buffer.read())
            
def load_checkpoint_from_gcs(gcs_path: str, model_class: pl.LightningModule):
    """
    Downloads a checkpoint from Google Cloud Storage (GCS) to a temporary file
    and loads the Lightning model.

    Args:
        gcs_path (str): The GCS path to the checkpoint (e.g., "gs://bucket/model/best.ckpt").
        model_class (pl.LightningModule): The class of the model (e.g., PytorchSoccerMapModel).

    Returns:
        pl.LightningModule: The loaded PyTorch Lightning model in evaluation mode.
    """

    # 1) Create a local temporary file
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=True) as tmp:

        # 2) Download from GCS to this temporary file
        fs = fsspec.filesystem("gs")
        with fs.open(gcs_path, "rb") as remote_file:
            tmp.write(remote_file.read())
            tmp.flush()

        # 3) Load the PL model
        model = model_class.load_from_checkpoint(tmp.name)
        model.eval()
        return model