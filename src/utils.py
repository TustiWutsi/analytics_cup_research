import pandas as pd
import gcsfs
import io
import pytorch_lightning as pl
from sklearn.pipeline import Pipeline
import joblib

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
            
def save_pipeline_gcs(fs: gcsfs.GCSFileSystem, gcs_path: str, pipeline: Pipeline):
    with fs.open(gcs_path, "wb") as f:
        joblib.dump(pipeline, f)
            
def load_pipeline_gcs(fs: gcsfs.GCSFileSystem,gcs_path: str) -> Pipeline:
    with fs.open(gcs_path, "rb") as f:
        return joblib.load(f)
    
def load_pipeline_local(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pipeline file not found locally at: {file_path}")
    pipeline = joblib.load(file_path)
    return pipeline
            
def load_checkpoint_from_gcs(gcs_path: str, model_class: pl.LightningModule):
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