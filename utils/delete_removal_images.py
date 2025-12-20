# Move images listed in Removal.csv to a backup folder
import shutil
from tqdm import tqdm

images_dir = Path(r'c:\Users\nikhi\Desktop\CXR_LT_ISBI\Dataset\images')
backup_dir = Path(r'c:\Users\nikhi\Desktop\CXR_LT_ISBI\Dataset\removed_images_backup')
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