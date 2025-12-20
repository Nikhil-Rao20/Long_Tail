import os
import pandas as pd
import numpy as np
import torch

from config import CLASS_NAMES, OUTPUT_DIR


def create_submission(image_ids, predictions, filename="submission.csv"):
    """Create submission CSV file."""
    submission_df = pd.DataFrame()
    submission_df["ImageID"] = image_ids
    
    for i, class_name in enumerate(CLASS_NAMES):
        submission_df[class_name] = predictions[:, i]
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    submission_df.to_csv(filepath, index=False)
    print(f"Submission saved: {filepath}")
    print(f"Shape: {submission_df.shape}")
    
    return submission_df


def validate_submission(submission_path):
    """Validate submission file format."""
    df = pd.read_csv(submission_path)
    
    # Check columns
    required_cols = ["ImageID"] + CLASS_NAMES
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False
    
    # Check ImageID is string
    if df["ImageID"].dtype != object:
        print("WARNING: ImageID should be string type")
    
    # Check predictions are in [0, 1]
    for col in CLASS_NAMES:
        if df[col].min() < 0 or df[col].max() > 1:
            print(f"ERROR: {col} has values outside [0, 1]")
            return False
    
    print("Submission validation PASSED!")
    return True


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
