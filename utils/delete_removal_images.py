# Move images listed in Removal.csv to a backup folder
import shutil
from tqdm import tqdm
import pandas as pd
removal_df = pd.read_csv(r'CXRLT-2026-TRAINING-DATA/Removal.csv')
from pathlib import Path
images_dir = Path(r'Dataset/images')
backup_dir = Path(r'Dataset/removed_images_backup')
backup_dir.mkdir(parents=True, exist_ok=True)

removal_ids = set(removal_df['ImageID'].values)

moved = 0
for img_name in tqdm(removal_ids, desc='Moving removed images'):
    src = images_dir / img_name
    dst = backup_dir / img_name
    if src.exists():
        shutil.move(str(src), str(dst))
        moved += 1

print(f"Moved {moved} images to backup folder: {backup_dir}")