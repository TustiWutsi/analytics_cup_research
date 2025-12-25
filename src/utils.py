import pandas as pd
import gcsfs
import io

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