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
from torch.optim.lr_scheduler import OneCycleLR

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
        # Squeeze the output to match the target shape
        outputs = outputs.squeeze(-1)  # [batch_size, pred_len, 1] -> [batch_size, pred_len]
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
            # Squeeze the output to match the target shape
            outputs = outputs.squeeze(-1) # [batch_size, pred_len, 1] -> [batch_size, pred_len]
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

def train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device, early_stopping_patience):
    """
    Trains the model using:
      - Pre-allocation of memory via a dummy pass.
      - OneCycleLR scheduler with warmup.
      - Gradient clipping.
      - Optionally, a robust loss such as SmoothL1Loss (Huber loss) if desired.
    """
    # -------------------------------
    # Preallocate memory for max input sizes
    # -------------------------------
    print("Preallocating memory using a dummy forward/backward pass...")
    # We get one batch, which should be close to the maximum expected size.
    for batch in train_dataloader:
        dummy_inputs, dummy_targets = batch
        dummy_inputs = dummy_inputs.to(device)
        dummy_targets = dummy_targets.to(device)
        dummy_output = model(dummy_inputs)
        # If the model output has a trailing singleton dimension, squeeze it.
        if dummy_output.dim() == 3 and dummy_output.size(-1) == 1:
            dummy_output = dummy_output.squeeze(-1)
        # Similarly, if dummy_targets has an extra dimension, squeeze it.
        if dummy_targets.dim() == 3 and dummy_targets.size(-1) == 1:
            dummy_targets = dummy_targets.squeeze(-1)
        expected_pred_len = dummy_output.size(1)
        if dummy_targets.size(1) != expected_pred_len:
            dummy_targets = dummy_targets[:, :expected_pred_len]
        dummy_loss = loss_function(dummy_output, dummy_targets)
        # Backward pass to preallocate internal buffers for backward
        dummy_loss.backward()
        optimizer.zero_grad()
        break
    print("Memory preallocation done.")
    
    # -------------------------------
    # Setup OneCycleLR Scheduler with warmup
    # -------------------------------
    scheduler = OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],  # Starting max learning rate
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        anneal_strategy='linear',
        pct_start=0.1  # First 10% as warmup
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0
        num_batches = 0
        total_grad_norm = 0.0
        
        for (inputs, targets) in tqdm(train_dataloader, desc=f"Epoch {epoch+1} progress", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # Squeeze the model output if necessary.
            if outputs.dim() == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            # Similarly, squeeze targets if they have an extra dimension.
            if targets.dim() == 3 and targets.size(-1) == 1:
                targets = targets.squeeze(-1)
            # If the target sequence length does not match the output, truncate targets.
            if targets.size(1) != outputs.size(1):
                targets = targets[:, :outputs.size(1)]
            loss = loss_function(outputs, targets)
            loss.backward()
            
            # Gradient Clipping: capture gradient norm
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm
            
            optimizer.step()
            scheduler.step()  # Updates learning rate according to OneCycle policy
            
            train_loss_accum += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss_accum / num_batches
        history['train_loss'].append(avg_train_loss)
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for (inputs, targets) in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # Squeeze output and targets if they have an extra dimension.
                if outputs.dim() == 3 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if targets.dim() == 3 and targets.size(-1) == 1:
                    targets = targets.squeeze(-1)
                # Truncate targets if necessary.
                if targets.size(1) != outputs.size(1):
                    targets = targets[:, :outputs.size(1)]
                loss = loss_function(outputs, targets)
                val_loss_accum += loss.item()
                val_batches += 1
                all_val_preds.append(outputs.detach().cpu())
                all_val_targets.append(targets.detach().cpu())
        avg_val_loss = val_loss_accum / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # Compute additional validation metrics: R² and RMSE
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        mse_val = torch.mean((all_val_preds - all_val_targets) ** 2).item()
        rmse_val = mse_val ** 0.5
        ss_res = torch.sum((all_val_targets - all_val_preds) ** 2)
        ss_tot = torch.sum((all_val_targets - torch.mean(all_val_targets)) ** 2)
        r2_val = 1 - (ss_res / ss_tot).item() if ss_tot != 0 else 0.0

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val R²: {r2_val:.4f} | Val RMSE: {rmse_val:.4f} | Avg Grad Norm: {avg_grad_norm:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history 