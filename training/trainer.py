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

# Initialize logging
logging.getLogger(__name__)

def train_epoch(model, dataloader, loss_function, optimizer, device, gradient_clip_value):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
        
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    avg_loss = epoch_loss / len(dataloader.dataset)
    return avg_loss

def validate_epoch(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    avg_loss = val_loss / len(dataloader.dataset)
    return avg_loss

def train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device, early_stopping_patience, gradient_clip_value=1.0):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_history = {'train_loss': [], 'val_loss': []}
    best_model_path = os.path.join("best_model.pth")
    
    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss = train_epoch(model, train_dataloader, loss_function, optimizer, device, gradient_clip_value)
        val_loss = validate_epoch(model, val_dataloader, loss_function, device)
        # If validation loss is NaN, set it to infinity for comparison purposes
        if math.isnan(val_loss):
            print(f"Epoch {epoch}: Validation loss is NaN. Setting to infinity for comparison.")
            val_loss = float('inf')
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
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