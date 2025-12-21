import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMAGE_DIR, TEST_IMAGE_DIR, TRAIN_CSV, TEST_CSV, CLASS_NAMES, 
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, VAL_SPLIT, RANDOM_SEED
)


class CXRDataset(Dataset):
    """Chest X-Ray Dataset for multi-label classification."""
    
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.class_names = CLASS_NAMES
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageID"]
        image_path = os.path.join(self.image_dir, image_id)
        
        # Load image as grayscale
        # Medical images are often 16-bit, need to normalize to 0-255
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        
        # Normalize 16-bit to 8-bit range (0-255)
        if image.max() > 255:
            image = (image / 65535.0 * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Replicate to 3 channels for pretrained models
        image = np.stack([image, image, image], axis=-1)  # (H, W, 3)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        if self.is_test:
            return image, image_id
        
        # Get labels
        labels = row[self.class_names].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels


def get_train_transforms():
    """Training augmentations."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        # A.OneOf([
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        #     A.RandomGamma(gamma_limit=(80, 120), p=1),
        # ], p=0.5),
        # A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Validation/Test transforms - no augmentation."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_and_split_data():
    """Load training data and split into train/val with patient-level grouping."""
    df = pd.read_csv(TRAIN_CSV)
    
    # Patient-level split using GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(df, groups=df["PatientID"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Unique patients in train: {train_df['PatientID'].nunique()}")
    print(f"Unique patients in val: {val_df['PatientID'].nunique()}")
    
    return train_df, val_df


def get_class_counts(df):
    """Get class counts for loss weighting."""
    counts = df[CLASS_NAMES].sum().values
    return counts


def get_pos_weights(df):
    """Calculate positive weights for BCEWithLogitsLoss."""
    counts = get_class_counts(df)
    total = len(df)
    pos_weights = (total - counts) / (counts + 1e-6)
    return torch.tensor(pos_weights, dtype=torch.float32)


def create_dataloaders(train_df, val_df):
    """Create train and validation dataloaders."""
    train_dataset = CXRDataset(
        train_df, IMAGE_DIR, transform=get_train_transforms(), is_test=False
    )
    val_dataset = CXRDataset(
        val_df, IMAGE_DIR, transform=get_val_transforms(), is_test=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader():
    """Create test dataloader for inference."""
    test_df = pd.read_csv(TEST_CSV)
    
    test_dataset = CXRDataset(
        test_df, TEST_IMAGE_DIR, transform=get_val_transforms(), is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return test_loader, test_df


if __name__ == "__main__":
    # Quick test
    train_df, val_df = load_and_split_data()
    print("\nClass distribution in training set:")
    print(train_df[CLASS_NAMES].sum().sort_values(ascending=False))
