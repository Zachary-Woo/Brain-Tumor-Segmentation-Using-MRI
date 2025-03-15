# Assignment 2: Deep Learning-based Brain Tumor Segmentation Using MRI
# Name: Zachary Wood
# Date: 3/12/2025

# Data Details:
# imagesTr is the folder containing the training samples
# labelsTr is the folder contianing segmentation masks for the training samples

# Task:
# 1. Data Visualization
#     - Use ITK-SNAP software to visualize a few MRI samples and their corresponding segmentation masks.
# 2. Brain tumor segmentation using U-Net
#     - Train 2D U-Net (process MRI slice by slice) or 3D U-Net (volumetric processing) for brain tumor segmentation. Do
#       5-fold cross validation on the training set ("imagesTr" folder) and report the segmentation results (Dice score and
#       Hausdorff Dist.) *****I am going with 2D U-Net*****
# 3. What to report
#     - Present a visualization example (data and segmentation mask) using the ITK-SNAP software or the code.
#     - Implementation details of the network.
#     - Use a table to list the segmentation results (Dice score and Hausdorff Dist.) for each fold, as well as the average
#       results of 5-fold cross validation.

# Code for Hausdorff Distance: https://loli.github.io/medpy/_modules/medpy/metric/binary.html#hd

# Evaluation:
# background (label 0), necrotic and non-enhancing tumor (label 1), peritumoral edema (label 2) and GD-enhancing
# tumor (label 4). The segmentation accuracy is measured by the Dice score and the Hausdorff distance (95%) metrics
# for enhancing tumor region (ET, label 4), regions of the tumor core (TC, labels 1 and 4), and the whole tumor region
# (WT, labels 1,2 and 4).

# 4. Present a few examples of your segmentation results (an example is given below) for qualitative analysis.

# What to Submit:
# 1. A report for this assignment. The quality of the report is important.
# 2. Clean code and clear instructions (e.g., a readme file) to reproduce your results. If you choose to host the code on
#    GitHub, please provide the GitHub link.

# Code Examples:
# 1. TransBTS for brain tumor segmentation: https://github.com/Wenxuan-1119/TransBTS
# 2. Brain tumor 3D segmentation with MONAI: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
# 3. MONAI Tutorials: https://github.com/Project-MONAI/tutorials

# Import necessary libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import generate_binary_structure, binary_erosion, distance_transform_edt
import random

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
data_root = "Task01_BrainTumour"
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

# Get list of all files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz') and not f.startswith('._')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz') and not f.startswith('._')])

# Make sure we're only using files that have both image and label
valid_files = []
for img_file in image_files:
    # Check if corresponding label file exists
    if img_file in label_files:
        valid_files.append(img_file)

# Use valid_files for both images and labels
image_files = valid_files
label_files = valid_files

# Define the U-Net model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # Output
        x = self.outconv(x)
        return x

# Define the dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform
        
        # Store slice information without loading data
        self.slices_info = []
        self._scan_data()
        
    def _scan_data(self):
        """Scan data to identify slices with tumors without loading entire volumes"""
        print("Scanning data to identify slices with tumors...")
        for img_path, label_path in tqdm(zip(self.image_files, self.label_files), total=len(self.image_files)):
            try:
                # Load label file to identify slices with tumors
                label_nib = nib.load(os.path.join(labels_dir, label_path))
                label_data = label_nib.get_fdata()
                
                # Get shape information
                img_nib = nib.load(os.path.join(images_dir, img_path))
                img_shape = img_nib.shape
                
                print(f"Scanning {img_path}: image shape {img_shape}, label shape {label_data.shape}")
                
                # Find slices with tumors
                for z in range(label_data.shape[2]):
                    # Check if this slice has tumor
                    if np.sum(label_data[:, :, z]) > 0:
                        # Store slice information without loading the actual data
                        self.slices_info.append({
                            'img_path': img_path,
                            'label_path': label_path,
                            'slice_idx': z,
                        })
            except Exception as e:
                print(f"Error scanning {img_path}: {e}")
        
        print(f"Found {len(self.slices_info)} slices with tumor regions")
    
    def _normalize(self, data):
        """Normalize data to [0, 1] range"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def __len__(self):
        return len(self.slices_info)
    
    def __getitem__(self, idx):
        # Get slice information
        slice_info = self.slices_info[idx]
        img_path = slice_info['img_path']
        label_path = slice_info['label_path']
        slice_idx = slice_info['slice_idx']
        
        try:
            # Load image and extract only the needed slice
            img_nib = nib.load(os.path.join(images_dir, img_path))
            img_data = img_nib.dataobj  # Using dataobj instead of get_fdata() to avoid loading entire volume
            
            # Extract the specific slice
            img_slice = np.array(img_data[:, :, slice_idx, :])  # Shape: [H, W, 4]
            
            # Normalize each modality
            for c in range(img_slice.shape[2]):
                img_slice[:, :, c] = self._normalize(img_slice[:, :, c])
            
            # Load label and extract only the needed slice
            label_nib = nib.load(os.path.join(labels_dir, label_path))
            label_data = label_nib.dataobj
            label_slice = np.array(label_data[:, :, slice_idx])  # Shape: [H, W]
            
            # Convert to tensors
            img = torch.tensor(img_slice, dtype=torch.float32).permute(2, 0, 1)  # [4, H, W]
            
            # Convert label to one-hot encoding
            label_tensor = torch.tensor(label_slice, dtype=torch.long)
            
            # Create one-hot encoding
            one_hot = torch.zeros((4, *label_tensor.shape), dtype=torch.float32)
            
            # Background (class 0)
            one_hot[0] = (label_tensor == 0).float()
            
            # Necrotic and non-enhancing tumor (class 1)
            one_hot[1] = (label_tensor == 1).float()
            
            # Peritumoral edema (class 2)
            one_hot[2] = (label_tensor == 2).float()
            
            # GD-enhancing tumor (class 4, mapped to index 3)
            one_hot[3] = (label_tensor == 4).float()
            
            # Apply transformations if specified
            if self.transform:
                img, one_hot = self.transform(img, one_hot)
            
            return img, one_hot, slice_idx
            
        except Exception as e:
            print(f"Error loading slice {slice_idx} from {img_path}: {e}")
            # Return a dummy sample in case of error
            # This prevents the dataloader from crashing
            dummy_img = torch.zeros((4, 240, 240), dtype=torch.float32)
            dummy_label = torch.zeros((4, 240, 240), dtype=torch.float32)
            dummy_label[0] = 1.0  # All background
            return dummy_img, dummy_label, slice_idx

# Define the Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        # Default class weights if none provided (giving higher weight to tumor classes)
        self.weight = weight if weight is not None else torch.tensor([0.1, 0.3, 0.3, 0.3])

    def forward(self, inputs, targets, smooth=1):
        # Apply softmax to get probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # Initialize total loss
        total_loss = 0
        
        # Calculate Dice loss for each class separately
        for i in range(inputs.size(1)):
            # Get current class
            input_class = inputs[:, i, :, :]
            target_class = targets[:, i, :, :]
            
            # Flatten inputs and targets, ensuring they are contiguous
            input_flat = input_class.contiguous().reshape(-1)
            target_flat = target_class.contiguous().reshape(-1)
            
            # Calculate intersection and dice score
            intersection = (input_flat * target_flat).sum()
            dice = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
            
            # Apply class weight and add to total loss
            total_loss += self.weight[i] * (1 - dice)
        
        # Return average loss
        return total_loss / inputs.size(1)

# Function to calculate Dice score
def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

# Function to calculate Hausdorff distance
def hausdorff_distance(pred, target):
    """
    Calculate the 95% Hausdorff Distance between binary objects.
    
    Parameters:
    -----------
    pred : torch.Tensor
        Prediction binary mask
    target : torch.Tensor
        Ground truth binary mask
        
    Returns:
    --------
    float
        The 95% Hausdorff distance
    """
    pred = pred.cpu().numpy().astype(bool)
    target = target.cpu().numpy().astype(bool)
    
    # Quick check for identical masks
    if np.array_equal(pred, target):
        # If masks are identical and contain positive values, return 0
        if np.any(pred):
            return 0.0
        # If masks are identical but empty, return 0
        else:
            return 0.0
    
    # Check if both arrays have positive values
    if np.any(pred) and np.any(target):
        try:
            # Use medpy's implementation directly for more reliable results
            try:
                from medpy.metric.binary import hd95
                return hd95(pred, target)
            except Exception as e:
                print(f"Error using medpy.metric.binary.hd95: {e}")
                
                # Fallback to our own implementation
                # Generate binary structure for surface extraction
                footprint = generate_binary_structure(pred.ndim, 1)
                
                # Extract surface voxels
                pred_border = pred ^ binary_erosion(pred, structure=footprint, iterations=1)
                target_border = target ^ binary_erosion(target, structure=footprint, iterations=1)
                
                # Check if borders are empty
                if not np.any(pred_border) or not np.any(target_border):
                    print("Warning: Empty borders detected in Hausdorff calculation")
                    return 150.0
                
                # Compute distance transforms
                dt_pred = distance_transform_edt(~pred_border)
                dt_target = distance_transform_edt(~target_border)
                
                # Get surface distances in both directions
                sds_pred = dt_target[pred_border]
                sds_target = dt_pred[target_border]
                
                # Combine distances and calculate 95th percentile
                all_distances = np.hstack((sds_pred, sds_target))
                if len(all_distances) > 0:
                    return np.percentile(all_distances, 95)
                else:
                    print("Warning: No distances calculated in Hausdorff distance")
                    return 150.0  # Return high value if no distances calculated
        except Exception as e:
            print(f"Error calculating Hausdorff distance: {e}")
            return 150.0  # Return high value on error
    elif np.any(target):  # Ground truth has tumor but prediction doesn't
        return 150.0  # Penalize with high value
    elif np.any(pred):  # Prediction has tumor but ground truth doesn't
        return 150.0  # Penalize with high value
    else:  # Both are empty
        return 0.0  # Both are empty, so distance is 0

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    """
    Train the model.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    num_epochs : int
        Number of epochs to train for
    checkpoint_path : str
        Path to save the best model checkpoint
        
    Returns:
    --------
    tuple
        (train_losses, val_losses)
    """
    # Check if loaders have data
    if len(train_loader) == 0:
        raise ValueError("Training data loader is empty. Cannot train the model.")
    
    if len(val_loader) == 0:
        raise ValueError("Validation data loader is empty. Cannot validate the model.")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Calculate average training loss for this epoch
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                epoch_val_loss += loss.item()
        
        # Calculate average validation loss for this epoch
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate based on validation loss
        prev_lr = current_lr
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change if it occurred
        if current_lr != prev_lr:
            print(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}')
    
    # Load the best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss {checkpoint['best_val_loss']:.4f}")
    
    return train_losses, val_losses

# Function to plot training and validation losses
def plot_losses(train_losses, val_losses, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses for Fold {fold+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_fold{fold+1}.png')
    plt.close()

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    
    # Initialize metrics
    dice_et = []
    dice_tc = []
    dice_wt = []
    hd95_et = []
    hd95_tc = []
    hd95_wt = []
    
    # Store some examples for visualization
    examples = []
    
    # Count slices with enhancing tumors
    et_count = 0
    et_pred_count = 0
    
    # Detailed ET statistics
    et_pixel_counts_gt = []
    et_pixel_counts_pred = []
    et_perfect_matches = 0
    et_zero_hd_count = 0
    
    with torch.no_grad():
        for images, labels, slice_idx in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            preds = torch.argmax(probs, dim=1)
            
            # Store a few examples for later visualization
            if len(examples) < 5:
                examples.append((images.cpu(), labels.cpu(), preds.cpu(), slice_idx))
            
            # Convert predictions to binary masks for each region
            for i in range(images.size(0)):
                # Ground truth - labels are already one-hot encoded
                # Channel 0: background (label 0)
                # Channel 1: necrotic and non-enhancing tumor (label 1)
                # Channel 2: peritumoral edema (label 2)
                # Channel 3: enhancing tumor (label 4)
                gt_et = labels[i, 3].float()  # Enhancing tumor (label 4)
                
                # For tumor core, combine necrotic and enhancing tumor using logical OR
                # Convert to boolean, perform logical OR, then back to float
                gt_tc = ((labels[i, 1] > 0.5) | (labels[i, 3] > 0.5)).float()  # Tumor core (labels 1 and 4)
                
                # For whole tumor, combine all tumor classes
                gt_wt = ((labels[i, 1] > 0.5) | (labels[i, 2] > 0.5) | (labels[i, 3] > 0.5)).float()  # Whole tumor (labels 1, 2, and 4)
                
                # Predictions - convert from class indices to binary masks
                # preds has shape [batch_size, H, W] with values 0-3
                # 0: background, 1: necrotic, 2: edema, 3: enhancing
                pred_et = (preds[i] == 3).float()  # Enhancing tumor (index 3 for class 4)
                pred_tc = ((preds[i] == 1) | (preds[i] == 3)).float()  # Tumor core (indices 1 and 3)
                pred_wt = ((preds[i] == 1) | (preds[i] == 2) | (preds[i] == 3)).float()  # Whole tumor (indices 1, 2, and 3)
                
                # Count slices with enhancing tumors
                gt_et_pixels = torch.sum(gt_et).item()
                pred_et_pixels = torch.sum(pred_et).item()
                
                if gt_et_pixels > 0:
                    et_count += 1
                    et_pixel_counts_gt.append(gt_et_pixels)
                
                if pred_et_pixels > 0:
                    et_pred_count += 1
                    et_pixel_counts_pred.append(pred_et_pixels)
                
                # Check for perfect matches in ET
                if gt_et_pixels > 0 and torch.all(gt_et == pred_et):
                    et_perfect_matches += 1
                
                # Calculate Dice scores
                dice_et.append(dice_score(pred_et, gt_et))
                dice_tc.append(dice_score(pred_tc, gt_tc))
                dice_wt.append(dice_score(pred_wt, gt_wt))
                
                # Calculate Hausdorff distances
                # For enhancing tumor
                if gt_et_pixels > 0:
                    hd_et = hausdorff_distance(pred_et, gt_et)
                    hd95_et.append(hd_et)
                    
                    # Count zero HD cases
                    if hd_et == 0 and pred_et_pixels > 0:
                        et_zero_hd_count += 1
                        print(f"Warning: HD95_ET is 0 for a slice with ET present. GT pixels: {gt_et_pixels}, Pred pixels: {pred_et_pixels}")
                        
                        # Additional check: are the masks identical?
                        if not torch.all(gt_et == pred_et):
                            print(f"  Unusual: HD95=0 but masks are not identical. This suggests an issue with HD calculation.")
                
                # For tumor core
                if torch.any(gt_tc):
                    hd95_tc.append(hausdorff_distance(pred_tc, gt_tc))
                
                # For whole tumor
                if torch.any(gt_wt):
                    hd95_wt.append(hausdorff_distance(pred_wt, gt_wt))
    
    # Print diagnostic information
    print(f"\n===== ENHANCING TUMOR (ET) DIAGNOSTICS =====")
    print(f"Total slices with ET in ground truth: {et_count}")
    print(f"Total slices with ET in predictions: {et_pred_count}")
    
    if et_count > 0:
        print(f"Average ET pixels per slice (ground truth): {np.mean(et_pixel_counts_gt):.2f}")
    if et_pred_count > 0:
        print(f"Average ET pixels per slice (prediction): {np.mean(et_pixel_counts_pred):.2f}")
    
    print(f"Perfect ET mask matches: {et_perfect_matches}")
    print(f"Zero Hausdorff distance cases: {et_zero_hd_count}")
    
    if et_zero_hd_count > 0 and et_perfect_matches < et_zero_hd_count:
        print("WARNING: There are cases with zero Hausdorff distance but imperfect mask matches.")
        print("This strongly suggests an issue with the Hausdorff distance calculation.")
    
    # Calculate mean metrics
    mean_dice_et = np.mean(dice_et) if dice_et else 0
    mean_dice_tc = np.mean(dice_tc) if dice_tc else 0
    mean_dice_wt = np.mean(dice_wt) if dice_wt else 0
    
    mean_hd95_et = np.mean(hd95_et) if hd95_et else 0
    mean_hd95_tc = np.mean(hd95_tc) if hd95_tc else 0
    mean_hd95_wt = np.mean(hd95_wt) if hd95_wt else 0
    
    # Print summary of metrics
    print(f"\n===== METRICS SUMMARY =====")
    print(f"ET metrics - Dice: {mean_dice_et:.4f}, HD95: {mean_hd95_et:.4f}, Samples: {len(hd95_et)}")
    print(f"TC metrics - Dice: {mean_dice_tc:.4f}, HD95: {mean_hd95_tc:.4f}, Samples: {len(hd95_tc)}")
    print(f"WT metrics - Dice: {mean_dice_wt:.4f}, HD95: {mean_hd95_wt:.4f}, Samples: {len(hd95_wt)}")
    
    return {
        'dice_et': mean_dice_et,
        'dice_tc': mean_dice_tc,
        'dice_wt': mean_dice_wt,
        'hd95_et': mean_hd95_et,
        'hd95_tc': mean_hd95_tc,
        'hd95_wt': mean_hd95_wt
    }, examples

# Function to visualize results
def visualize_results(examples, fold, output_dir):
    """
    Visualize segmentation results.
    
    Parameters:
    -----------
    examples : list
        List of tuples containing (images, labels, predictions, slice_idx)
    fold : int
        Fold number
    output_dir : str
        Output directory
    """
    plt.figure(figsize=(15, 10))
    
    for i, (images, labels, preds, slice_idx) in enumerate(examples[:5]):  # Show up to 5 examples
        if i >= 5:  # Limit to 5 examples
            break
            
        # Get the first image in the batch
        image = images[0]  # Get first image from batch
        label = labels[0]  # Get first label from batch
        pred = preds[0]    # Get first prediction from batch
        
        # Extract T1ce modality for visualization
        t1ce = image[0].numpy()
        
        # Convert to binary masks
        # Labels are already one-hot encoded
        # Channel 0: background (label 0)
        # Channel 1: necrotic and non-enhancing tumor (label 1)
        # Channel 2: peritumoral edema (label 2)
        # Channel 3: enhancing tumor (label 4)
        gt_et = label[3].float().numpy()  # Enhancing tumor (label 4)
        
        # Use logical operations for combining masks
        gt_tc = ((label[1] > 0.5) | (label[3] > 0.5)).float().numpy()  # Tumor core (labels 1 and 4)
        gt_wt = ((label[1] > 0.5) | (label[2] > 0.5) | (label[3] > 0.5)).float().numpy()  # Whole tumor (labels 1, 2, and 4)
        
        # Predictions - convert from class indices to binary masks
        # preds has values 0-3 (0: background, 1: necrotic, 2: edema, 3: enhancing)
        pred_et = (pred == 3).float().numpy()  # Enhancing tumor (index 3 for class 4)
        pred_tc = ((pred == 1) | (pred == 3)).float().numpy()  # Tumor core (indices 1 and 3)
        pred_wt = ((pred == 1) | (pred == 2) | (pred == 3)).float().numpy()  # Whole tumor (indices 1, 2, and 3)
        
        # Create a 3x3 grid for each example
        plt.subplot(5, 3, i*3 + 1)
        plt.imshow(t1ce, cmap='gray')
        plt.contour(gt_wt, colors='g', linewidths=0.5)
        plt.contour(gt_tc, colors='b', linewidths=0.5)
        plt.contour(gt_et, colors='r', linewidths=0.5)
        plt.title(f'Ground Truth (Slice {slice_idx})')
        plt.axis('off')
        
        plt.subplot(5, 3, i*3 + 2)
        plt.imshow(t1ce, cmap='gray')
        plt.contour(pred_wt, colors='g', linewidths=0.5)
        plt.contour(pred_tc, colors='b', linewidths=0.5)
        plt.contour(pred_et, colors='r', linewidths=0.5)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(5, 3, i*3 + 3)
        # Create a color-coded segmentation mask
        gt_mask = np.zeros((t1ce.shape[0], t1ce.shape[1], 3))
        pred_mask = np.zeros((t1ce.shape[0], t1ce.shape[1], 3))
        
        # Ground truth: Red = ET, Blue = TC, Green = WT
        gt_mask[gt_wt > 0.5, 1] = 1  # Green for WT
        gt_mask[gt_tc > 0.5, 2] = 1  # Blue for TC
        gt_mask[gt_et > 0.5, 0] = 1  # Red for ET
        
        # Prediction: Red = ET, Blue = TC, Green = WT
        pred_mask[pred_wt > 0.5, 1] = 1  # Green for WT
        pred_mask[pred_tc > 0.5, 2] = 1  # Blue for TC
        pred_mask[pred_et > 0.5, 0] = 1  # Red for ET
        
        # Overlay ground truth and prediction
        overlay = np.zeros((t1ce.shape[0], t1ce.shape[1], 3))
        overlay[..., 0] = np.maximum(gt_mask[..., 0] * 0.5, pred_mask[..., 0])  # Red channel
        overlay[..., 1] = np.maximum(gt_mask[..., 1] * 0.5, pred_mask[..., 1])  # Green channel
        overlay[..., 2] = np.maximum(gt_mask[..., 2] * 0.5, pred_mask[..., 2])  # Blue channel
        
        plt.imshow(overlay)
        plt.title('Overlay (GT + Pred)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', f'segmentation_results_fold{fold+1}.png'))
    plt.close()

# Define data augmentation transforms
def get_transforms():
    """
    Get data augmentation transforms for 2D slices.
    
    Returns:
    --------
    callable
        A function that applies transforms to both image and label
    """
    def apply_transforms(image, label):
        # Apply random rotation
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = torch.rot90(image, k=angle // 90, dims=[1, 2])
            label = torch.rot90(label, k=angle // 90, dims=[1, 2])
        
        # Apply random horizontal flip
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        
        # Apply random vertical flip
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])
        
        # Apply random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)
        
        return image, label
    
    return apply_transforms

# Function to analyze dataset for enhancing tumor regions
def analyze_dataset_et(data_loader):
    """
    Analyze the dataset to understand the distribution of enhancing tumor regions.
    
    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader containing the dataset to analyze
    """
    print("\n===== DATASET ANALYSIS FOR ENHANCING TUMOR REGIONS =====")
    
    total_slices = 0
    slices_with_et = 0
    et_pixel_counts = []
    et_pixel_percentages = []
    
    for images, labels, slice_idx in tqdm(data_loader, desc='Analyzing dataset'):
        total_slices += images.size(0)
        
        for i in range(images.size(0)):
            # Extract enhancing tumor mask (channel 3 for class 4)
            et_mask = labels[i, 3]  # Already a float tensor
            
            # Count pixels
            et_pixels = torch.sum(et_mask).item()
            total_pixels = et_mask.numel()
            
            if et_pixels > 0:
                slices_with_et += 1
                et_pixel_counts.append(et_pixels)
                et_pixel_percentages.append(100 * et_pixels / total_pixels)
    
    # Print statistics
    print(f"Total slices analyzed: {total_slices}")
    print(f"Slices containing enhancing tumor: {slices_with_et} ({100 * slices_with_et / total_slices:.2f}%)")
    
    if slices_with_et > 0:
        print(f"Average ET pixels per slice (when present): {np.mean(et_pixel_counts):.2f}")
        print(f"Median ET pixels per slice (when present): {np.median(et_pixel_counts):.2f}")
        print(f"Min ET pixels per slice: {np.min(et_pixel_counts):.2f}")
        print(f"Max ET pixels per slice: {np.max(et_pixel_counts):.2f}")
        print(f"Average percentage of slice covered by ET: {np.mean(et_pixel_percentages):.4f}%")
    
    print("==========================================================")
    
    return slices_with_et / total_slices

# Main function
def main():
    # Ask user if they want to train a new model or use a pre-trained model
    while True:
        choice = input("Do you want to train a new model or use a pre-trained model? (train/pretrained): ").strip().lower()
        if choice in ['train', 'pretrained']:
            break
        print("Invalid choice. Please enter 'train' or 'pretrained'.")
    
    # Set hyperparameters
    batch_size = 4  # Reduced batch size to save memory
    num_epochs = 10
    learning_rate = 0.001
    
    # Define paths
    data_dir = 'Task01_BrainTumour'
    output_dir = 'results_2d_unet'
    
    # Create output directories
    print(f"Creating output directories in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Define directories
    global images_dir, labels_dir
    images_dir = os.path.join(data_dir, 'imagesTr')
    labels_dir = os.path.join(data_dir, 'labelsTr')
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Get all image and label files (just the filenames, not full paths)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz') and not f.startswith('._')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz') and not f.startswith('._')])
    
    print(f"Found {len(image_files)} image files and {len(label_files)} label files")
    
    # Ensure matching files
    valid_files = []
    for img_file in image_files:
        # Check if corresponding label file exists
        if img_file in label_files:
            valid_files.append(img_file)
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    
    # Use valid_files for both images and labels
    image_files = valid_files
    label_files = valid_files
    
    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize results dictionary
    results = {
        'fold': [],
        'dice_et': [],
        'dice_tc': [],
        'dice_wt': [],
        'hd95_et': [],
        'hd95_tc': [],
        'hd95_wt': []
    }
    
    # Define transformations
    transform = get_transforms()
    
    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        print(f"\nTraining fold {fold+1}/5")
        
        # Split data
        train_img_files = [image_files[i] for i in train_idx]
        train_label_files = [label_files[i] for i in train_idx]
        val_img_files = [image_files[i] for i in val_idx]
        val_label_files = [label_files[i] for i in val_idx]
        
        # Create datasets
        train_dataset = BrainTumorDataset(train_img_files, train_label_files, transform=transform)
        val_dataset = BrainTumorDataset(val_img_files, val_label_files)
        
        # Check if datasets have data
        if len(train_dataset) == 0:
            print("ERROR: Training dataset is empty. Cannot proceed with training.")
            continue
        
        if len(val_dataset) == 0:
            print("ERROR: Validation dataset is empty. Cannot proceed with training.")
            continue
        
        print(f"Training dataset size: {len(train_dataset)} slices")
        print(f"Validation dataset size: {len(val_dataset)} slices")
        
        # Create data loaders with memory-efficient settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # No parallel workers to save memory
            pin_memory=False  # Don't pin memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
        
        # Try to free up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Analyze dataset for enhancing tumor regions
        if fold == 0:  # Only do this for the first fold to save time
            print("\nAnalyzing validation dataset for enhancing tumor regions...")
            try:
                et_prevalence = analyze_dataset_et(val_loader)
                if et_prevalence < 0.05:
                    print("\nWARNING: Very few enhancing tumor regions in the dataset.")
                    print("This may lead to unreliable metrics for the enhancing tumor class.")
            except Exception as e:
                print(f"Error during dataset analysis: {e}")
        
        # Initialize model
        model = UNet(in_channels=4, out_channels=4).to(device)
        
        # Define loss function with class weights
        class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(device)  # Lower weight for background
        criterion = DiceLoss(weight=class_weights)
        
        # Define optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Define checkpoint path
        checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_fold{fold+1}.pth')
        
        # Train or load model
        try:
            if choice == 'train':
                # Train the model
                train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path)
                
                # Plot training and validation losses
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Losses - Fold {fold+1}')
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'visualizations', f'loss_plot_fold{fold+1}.png'))
                plt.close()
            else:
                # Load pre-trained model
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded pre-trained model from {checkpoint_path}")
                else:
                    print(f"No pre-trained model found at {checkpoint_path}. Training a new model.")
                    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path)
            
            # Evaluate the model
            metrics, examples = evaluate_model(model, val_loader)
            
            # Store results
            results['fold'].append(fold+1)
            results['dice_et'].append(metrics['dice_et'])
            results['dice_tc'].append(metrics['dice_tc'])
            results['dice_wt'].append(metrics['dice_wt'])
            results['hd95_et'].append(metrics['hd95_et'])
            results['hd95_tc'].append(metrics['hd95_tc'])
            results['hd95_wt'].append(metrics['hd95_wt'])
            
            # Visualize results
            visualize_results(examples, fold, output_dir)
            
            # Print results for this fold
            print(f"\nResults for fold {fold+1}:")
            print(f"Dice ET: {metrics['dice_et']:.4f}")
            print(f"Dice TC: {metrics['dice_tc']:.4f}")
            print(f"Dice WT: {metrics['dice_wt']:.4f}")
            print(f"HD95 ET: {metrics['hd95_et']:.4f}")
            print(f"HD95 TC: {metrics['hd95_tc']:.4f}")
            print(f"HD95 WT: {metrics['hd95_wt']:.4f}")
            
            # Try to free up memory again
            del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during training/evaluation for fold {fold+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate average metrics
    if results['fold']:  # Only if we have results
        avg_dice_et = np.mean(results['dice_et'])
        avg_dice_tc = np.mean(results['dice_tc'])
        avg_dice_wt = np.mean(results['dice_wt'])
        avg_hd95_et = np.mean(results['hd95_et'])
        avg_hd95_tc = np.mean(results['hd95_tc'])
        avg_hd95_wt = np.mean(results['hd95_wt'])
        
        # Print overall results
        print("\nOverall Results:")
        print(f"Average Dice ET: {avg_dice_et:.4f}")
        print(f"Average Dice TC: {avg_dice_tc:.4f}")
        print(f"Average Dice WT: {avg_dice_wt:.4f}")
        print(f"Average HD95 ET: {avg_hd95_et:.4f}")
        print(f"Average HD95 TC: {avg_hd95_tc:.4f}")
        print(f"Average HD95 WT: {avg_hd95_wt:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        # Plot overall results
        plot_overall_results(results, output_dir)
    else:
        print("No results were collected. Training failed for all folds.")
    
    print(f"\nTraining and evaluation completed. Results saved to {output_dir}")

# Function to plot overall results across folds
def plot_overall_results(results, output_dir):
    """
    Plot overall results across folds.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each fold
    output_dir : str
        Output directory
    """
    # Extract data for plotting
    folds = results['fold']
    
    # Plot Dice scores
    plt.figure(figsize=(10, 6))
    plt.plot(folds, results['dice_et'], 'o-', label='Enhancing Tumor')
    plt.plot(folds, results['dice_tc'], 'o-', label='Tumor Core')
    plt.plot(folds, results['dice_wt'], 'o-', label='Whole Tumor')
    plt.xlabel('Fold')
    plt.ylabel('Dice Score')
    plt.title('Dice Scores Across Folds')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'dice_scores.png'))
    plt.close()
    
    # Plot Hausdorff distances
    plt.figure(figsize=(10, 6))
    plt.plot(folds, results['hd95_et'], 'o-', label='Enhancing Tumor')
    plt.plot(folds, results['hd95_tc'], 'o-', label='Tumor Core')
    plt.plot(folds, results['hd95_wt'], 'o-', label='Whole Tumor')
    plt.xlabel('Fold')
    plt.ylabel('Hausdorff Distance (95%)')
    plt.title('Hausdorff Distances Across Folds')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.savefig(os.path.join(output_dir, 'visualizations', 'hausdorff_distances.png'))
    plt.close()

if __name__ == "__main__":
    main()

