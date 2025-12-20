import os
import time
import torch
import numpy as np
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_DIR
from metrics import AverageMeter, compute_metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler=None, epoch=0):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    if scheduler is not None:
        scheduler.step()
        
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    
    all_labels = []
    all_preds = []
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), images.size(0))
        
        preds = torch.sigmoid(logits)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """Save model checkpoint."""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, metrics


def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler=None, 
    num_epochs=30,
    early_stopping_patience=5,
    model_name="model"
):
    """Full training loop."""
    model = model.to(DEVICE)
    
    best_mAP = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'mAP': [], 'mAUC': [], 'mF1': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion)
        
        # Log metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"mAP: {val_metrics['mAP']:.4f} | mAUC: {val_metrics['mAUC']:.4f} | mF1: {val_metrics['mF1']:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['mAP'].append(val_metrics['mAP'])
        history['mAUC'].append(val_metrics['mAUC'])
        history['mF1'].append(val_metrics['mF1'])
        
        # Save best model
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                f"{model_name}_best.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Save latest
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            f"{model_name}_latest.pth"
        )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
            
    print(f"\nTraining complete! Best mAP: {best_mAP:.4f}")
    return history, best_mAP


@torch.no_grad()
def predict(model, test_loader):
    """Generate predictions for test set."""
    model.eval()
    model = model.to(DEVICE)
    
    all_preds = []
    all_image_ids = []
    
    for images, image_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        logits = model(images)
        preds = torch.sigmoid(logits)
        
        all_preds.append(preds.cpu().numpy())
        all_image_ids.extend(image_ids)
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_image_ids, all_preds
