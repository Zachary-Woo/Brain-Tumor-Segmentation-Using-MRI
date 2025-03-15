import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils import dice_coefficient, hausdorff_distance

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_dir='checkpoints'):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        device (torch.device): Device to use for training
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
        
    Returns:
        tuple: Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        print(f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Check for NaN values
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or Inf in model outputs at batch {batch_idx}")
                continue
                
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(f"Warning: NaN or Inf in labels at batch {batch_idx}")
                continue
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss at batch {batch_idx}, skipping")
                continue
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate Dice score
            with torch.no_grad():
                # Get predicted class
                pred = torch.argmax(outputs, dim=1, keepdim=True)
                target = torch.argmax(labels, dim=1, keepdim=True)
                
                # Calculate Dice score
                dice = (2.0 * (pred * target).sum()) / (pred.sum() + target.sum() + 1e-5)
                train_dice += dice.item()
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation phase
        val_loss, val_dice, val_metrics = evaluate_model(model, val_loader, device, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        # Print metrics
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    return model, history

def evaluate_model(model, val_loader, device, criterion=None):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for evaluation
        criterion (nn.Module, optional): Loss function. Defaults to None.
        
    Returns:
        tuple: Validation loss and Dice score
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    # Counters for metrics
    total_samples = 0
    valid_samples = 0
    
    # Metrics for each class
    class_dice = np.zeros(4)  # Background, Necrotic, Edema, Enhancing
    class_hausdorff = np.zeros(4)
    class_samples = np.zeros(4)
    
    # Store examples for visualization
    examples = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            
            # Get predicted class
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            target = torch.argmax(labels, dim=1, keepdim=True)
            
            # Store examples for visualization (max 5)
            if len(examples) < 5:
                for i in range(min(images.size(0), 5 - len(examples))):
                    examples.append({
                        'image': images[i].detach().cpu(),
                        'ground_truth': target[i].detach().cpu(),
                        'prediction': pred[i].detach().cpu()
                    })
            
            # Calculate Dice score
            dice = (2.0 * (pred * target).sum()) / (pred.sum() + target.sum() + 1e-5)
            val_dice += dice.item()
            
            # Update counters
            total_samples += images.size(0)
            
            # Calculate metrics for each class
            for c in range(4):  # 0: Background, 1: Necrotic, 2: Edema, 3: Enhancing
                # Create binary masks for this class
                pred_c = (pred == c).float()
                target_c = (target == c).float()
                
                # Check if this class exists in the ground truth
                if target_c.sum() > 0:
                    # Update class samples counter
                    class_samples[c] += 1
                    
                    # Calculate Dice for this class
                    dice_c = (2.0 * (pred_c * target_c).sum()) / (pred_c.sum() + target_c.sum() + 1e-5)
                    class_dice[c] += dice_c.item()
                    
                    # For Hausdorff, we need to process each slice separately
                    for i in range(images.size(0)):
                        pred_slice = pred_c[i].squeeze().cpu().numpy()
                        target_slice = target_c[i].squeeze().cpu().numpy()
                        
                        # Only calculate if both have positive pixels
                        if np.sum(pred_slice) > 0 and np.sum(target_slice) > 0:
                            try:
                                from medpy.metric.binary import hd95
                                hd = hd95(pred_slice, target_slice)
                                if not np.isnan(hd) and not np.isinf(hd):
                                    class_hausdorff[c] += hd
                                    valid_samples += 1
                            except Exception as e:
                                print(f"Error computing Hausdorff distance: {e}")
    
    # Calculate average metrics
    if len(val_loader) > 0:
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Calculate average metrics for each class
        for c in range(4):
            if class_samples[c] > 0:
                class_dice[c] /= class_samples[c]
                class_hausdorff[c] /= max(1, valid_samples)  # Avoid division by zero
                
                print(f"Class {c} - Dice: {class_dice[c]:.4f}, "
                      f"Hausdorff: {class_hausdorff[c]:.4f}, "
                      f"Samples: {int(class_samples[c])}")
            else:
                print(f"Class {c} - No valid samples")
    
    # Return metrics and examples
    metrics = {
        'loss': val_loss,
        'dice': val_dice,
        'class_dice': class_dice,
        'class_hausdorff': class_hausdorff,
        'examples': examples
    }
    
    return val_loss, val_dice, metrics 