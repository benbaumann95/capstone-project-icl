import pandas as pd
from pathlib import Path
import datetime

def load_data(func_id):
    """Loads the samples.csv for a specific function."""
    # Determine the project root relative to this file
    # This file is in src/utils.py, so root is parent of parent
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / f"function_{func_id}" / "samples.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find samples file at {csv_path}")
    return pd.read_csv(csv_path)

def save_submission(func_id, query, module_name="Unknown"):
    """Saves the submission to a log file."""
    project_root = Path(__file__).resolve().parent.parent
    submission_dir = project_root / "submissions"
    submission_dir.mkdir(exist_ok=True)
    
    log_path = submission_dir / "submission_log.csv"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to write header
    file_exists = log_path.exists()
    
    with open(log_path, "a") as f:
        if not file_exists:
            f.write("timestamp,module,function_id,query\n")
        f.write(f"{timestamp},{module_name},{func_id},{query}\n")
    
    print(f"Saved submission for Function {func_id} to {log_path}")
