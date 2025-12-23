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
    # model.load_state_dict(checkpoint, strict=False)
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
    model_name="model",
    sampler=None
):
    """Full training loop with support for DRW and progressive sampling."""
    model = model.to(DEVICE)
    
    best_mAP = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'mAP': [], 'mAUC': [], 'mF1': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Update DRW loss epoch if applicable
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
            if hasattr(criterion, 'drw_epoch') and epoch == criterion.drw_epoch:
                print(">>> DRW: Switching to class-balanced weights")
        
        # Update progressive sampler if applicable
        if sampler is not None and hasattr(sampler, 'update_epoch'):
            sampler.update_epoch(epoch)
        
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


def train_two_stage(
    model,
    train_loader,
    val_loader,
    train_loader_balanced,
    criterion_stage1,
    criterion_stage2,
    optimizer,
    scheduler=None,
    stage1_epochs=20,
    stage2_epochs=10,
    early_stopping_patience=5,
    model_name="model"
):
    """
    Two-stage training: cRT (classifier re-training).
    Stage 1: Train full model with standard loss
    Stage 2: Freeze backbone, retrain classifier with balanced loss/sampling
    """
    model = model.to(DEVICE)
    
    print("="*60)
    print("STAGE 1: Full model training")
    print("="*60)
    
    best_mAP = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'mAP': [], 'mAUC': [], 'mF1': [], 'stage': []}
    
    # Stage 1: Train full model
    for epoch in range(1, stage1_epochs + 1):
        print(f"\n[Stage 1] Epoch {epoch}/{stage1_epochs}")
        
        if hasattr(criterion_stage1, 'update_epoch'):
            criterion_stage1.update_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, criterion_stage1, optimizer, scheduler, epoch)
        val_metrics = validate(model, val_loader, criterion_stage1)
        
        print(f"Train Loss: {train_loss:.4f} | Val mAP: {val_metrics['mAP']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['mAP'].append(val_metrics['mAP'])
        history['mAUC'].append(val_metrics['mAUC'])
        history['mF1'].append(val_metrics['mF1'])
        history['stage'].append(1)
        
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            save_checkpoint(model, optimizer, epoch, val_metrics, f"{model_name}_stage1_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping Stage 1 at epoch {epoch}")
            break
    
    # Load best Stage 1 model
    load_checkpoint(model, None, os.path.join(CHECKPOINT_DIR, f"{model_name}_stage1_best.pth"))
    
    print("\n" + "="*60)
    print("STAGE 2: Classifier re-training (backbone frozen)")
    print("="*60)
    
    # Freeze backbone
    model.freeze_backbone()
    
    # New optimizer for classifier only
    classifier_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stage2 = torch.optim.AdamW(classifier_params, lr=1e-3, weight_decay=1e-4)
    
    patience_counter = 0
    stage2_best_mAP = best_mAP
    
    for epoch in range(1, stage2_epochs + 1):
        print(f"\n[Stage 2] Epoch {epoch}/{stage2_epochs}")
        
        if hasattr(criterion_stage2, 'update_epoch'):
            criterion_stage2.update_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader_balanced, criterion_stage2, optimizer_stage2, None, epoch)
        val_metrics = validate(model, val_loader, criterion_stage2)
        
        print(f"Train Loss: {train_loss:.4f} | Val mAP: {val_metrics['mAP']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['mAP'].append(val_metrics['mAP'])
        history['mAUC'].append(val_metrics['mAUC'])
        history['mF1'].append(val_metrics['mF1'])
        history['stage'].append(2)
        
        if val_metrics['mAP'] > stage2_best_mAP:
            stage2_best_mAP = val_metrics['mAP']
            save_checkpoint(model, optimizer_stage2, epoch, val_metrics, f"{model_name}_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping Stage 2 at epoch {epoch}")
            break
    
    # Unfreeze for future use
    model.unfreeze_backbone()
    
    print(f"\nTwo-stage training complete!")
    print(f"Stage 1 best mAP: {best_mAP:.4f}")
    print(f"Stage 2 best mAP: {stage2_best_mAP:.4f}")
    
    return history, stage2_best_mAP


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
