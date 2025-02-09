#!/usr/bin/env python
"""
Training and Validation Functions

Provides:
  - train_epoch: Runs a training epoch.
  - validate_epoch: Evaluates the model on the validation set.
  - train_model: Runs the complete training loop with early stopping and saves the best model.
"""

import torch
from torch.nn.utils import clip_grad_norm_
import os
import logging
import math
from tqdm import tqdm

# Initialize logging
logging.getLogger(__name__)

def train_epoch(model, dataloader, loss_function, optimizer, device, gradient_clip_value, epoch):
    model.train()
    epoch_loss = 0.0
    processed_samples = 0
    total_grad_norm = 0.0
    num_batches = 0
    # Wrap the dataloader with tqdm to show progress for this epoch
    for batch in tqdm(dataloader, desc=f"Epoch {epoch} progress", unit="batch"):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Check if loss is NaN, and if so, skip this batch update with a warning.
        if math.isnan(loss.item()):
            print("Warning: NaN loss encountered in batch; skipping update for this batch.")
            optimizer.zero_grad()
            continue

        loss.backward()
        
        # Apply gradient clipping and compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
        total_grad_norm += grad_norm
        num_batches += 1

        optimizer.step()

        batch_size = inputs.size(0)
        epoch_loss += loss.item() * batch_size
        processed_samples += batch_size

    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    print(f"Average Gradient Norm for Epoch {epoch}: {avg_grad_norm:.4f}")

    if processed_samples == 0:
        avg_loss = float('nan')
    else:
        avg_loss = epoch_loss / processed_samples
    return avg_loss

def validate_epoch(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
    avg_loss = val_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return avg_loss, r2.item()

def train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device, early_stopping_patience, gradient_clip_value=1.0):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_model_path = os.path.join("best_model.pth")
    
    # Create a ReduceLROnPlateau scheduler which reduces LR by factor 0.5 if the validation loss does not improve for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss = train_epoch(model, train_dataloader, loss_function, optimizer, device, gradient_clip_value, epoch)
        val_loss, r2 = validate_epoch(model, val_dataloader, loss_function, device)
        # If validation loss is NaN, set it to infinity for comparison purposes
        if math.isnan(val_loss):
            print(f"Epoch {epoch}: Validation loss is NaN. Setting to infinity for comparison.")
            val_loss = float('inf')
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history.setdefault('val_r2', []).append(r2)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Validation R2 = {r2:.4f}")
        
        # Step the scheduler with the current validation loss
        scheduler.step(val_loss)
        # Print current learning rate for monitoring
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        
        # Always save the model after the first epoch, then update only if loss improves
        if epoch == 1 or val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, training_history 