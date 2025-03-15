# Assignment 2: Deep Learning-based Brain Tumor Segmentation Using MRI
# Name: Zachary Wood
# Date: 3/12/2025 (Recent Modifications have broken this. V3 aims to fix this.)

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from monai.metrics import compute_hausdorff_distance
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    EnsureTyped,
    NormalizeIntensityd,
    CropForegroundd,
    Orientationd,
    Spacingd,
    RandCropByPosNegLabeld
)
from monai.losses import DiceLoss as DiceCELoss
from monai.data import Dataset as MonaiDataset
from monai.networks.nets import UNet as MonaiUNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import nibabel as nib
import random
from medpy.metric.binary import hd95

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Enable benchmark mode for faster training
    torch.backends.cudnn.benchmark = True
    # For numerical stability
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    def __init__(self, image_files, label_files, transform=None, resize=True, target_size=(240, 240)):
        self.image_files = [os.path.join(images_dir, f) for f in image_files]
        self.label_files = [os.path.join(labels_dir, f) for f in label_files]
        self.resize = resize
        self.target_size = target_size
        self.visualize_samples = True  # Flag to control visualization
        
        # Define simplified transforms for loading and preprocessing 3D data
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
        
        # We'll apply simpler transforms before slicing the data to 2D
        self.transform = Compose(base_transforms)
        
        # Create data list for MONAI dataset
        self.data_dicts = [
            {"image": image_file, "label": label_file}
            for image_file, label_file in zip(self.image_files, self.label_files)
        ]
        
        # Initialize MONAI dataset with simpler transforms
        self.dataset = MonaiDataset(self.data_dicts, transform=self.transform)
        
        # Store slices for all volumes to create a flat dataset
        self.all_slices = []
        
        # Pre-process datasets to extract 2D slices from 3D volumes
        print("Extracting 2D slices from 3D volumes...")
        try:
            total_slices = 0
            tumor_slices = 0
            et_slices = 0  # Count slices with enhancing tumor
            
            # Add tqdm progress bar for volume processing
            for i in tqdm(range(len(self.dataset)), desc="Processing volumes"):
                volume = self.dataset[i]
                
                # Ensure we have a dictionary output
                if not isinstance(volume, dict):
                    print(f"Warning: Expected dict but got {type(volume)} at index {i}")
                    continue
                
                # Get 3D volumes
                img_volume = volume["image"]
                lbl_volume = volume["label"]
                
                # Check that we have proper tensors
                if not isinstance(img_volume, torch.Tensor) or not isinstance(lbl_volume, torch.Tensor):
                    print(f"Warning: Non-tensor data at index {i}. Image: {type(img_volume)}, Label: {type(lbl_volume)}")
                    continue
                
                # Debug: Check for enhancing tumor in this volume
                # IMPORTANT: In BraTS dataset, enhancing tumor is label 3 (not 4)
                has_et = torch.any(lbl_volume == 3).item()
                if has_et and i < 5:  # Only print for first few volumes
                    print(f"Volume {i} contains enhancing tumor (label 3)")
                    # Count how many voxels have enhancing tumor
                    et_voxels = torch.sum(lbl_volume == 3).item()
                    total_voxels = lbl_volume.numel()
                    print(f"  - ET voxels: {et_voxels} ({100 * et_voxels / total_voxels:.4f}% of volume)")
                
                # Extract 2D slices along the z-axis
                slices_z = img_volume.shape[-1]
                middle_start = max(0, (slices_z // 2) - 20)  # Focus on middle slices where tumor is likely
                middle_end = min(slices_z, (slices_z // 2) + 20)
                
                # Store info about slices processed
                volume_slices = 0
                volume_tumor_slices = 0
                volume_et_slices = 0
                
                # Store central slices (more likely to contain tumor)
                for z in range(middle_start, middle_end):
                    img_slice = img_volume[..., z]
                    lbl_slice = lbl_volume[..., z]
                    
                    volume_slices += 1
                    
                    # Check for enhancing tumor in this slice (label 3)
                    has_et_slice = torch.any(lbl_slice == 3).item()
                    if has_et_slice:
                        volume_et_slices += 1
                        # Print details for first few ET slices
                        if et_slices < 5:
                            et_pixels = torch.sum(lbl_slice == 3).item()
                            total_pixels = lbl_slice.numel()
                            print(f"Found ET slice in volume {i}, slice {z}: {et_pixels} pixels ({100 * et_pixels / total_pixels:.4f}%)")
                    
                    # Skip slices without any tumor
                    has_tumor = torch.sum(lbl_slice > 0).item() >= 10
                    if transform and not has_tumor:
                        continue  # Skip slices with little or no tumor during training
                    
                    if has_tumor:
                        volume_tumor_slices += 1
                        
                    # Resize slices if requested to save memory
                    if self.resize:
                        img_slice = self._resize_slice(img_slice, self.target_size)
                        lbl_slice = self._resize_slice(lbl_slice, self.target_size)
                    
                    # Store the slice
                    self.all_slices.append((img_slice, lbl_slice, z))
                
                # Update total counters
                total_slices += volume_slices
                tumor_slices += volume_tumor_slices
                et_slices += volume_et_slices
            
            # Print summary statistics
            print(f"Processed {len(self.dataset)} volumes with {total_slices} total slices")
            print(f"Found {tumor_slices} slices containing tumor ({100 * tumor_slices / max(total_slices, 1):.2f}%)")
            print(f"Found {et_slices} slices containing enhancing tumor ({100 * et_slices / max(total_slices, 1):.2f}%)")
            print(f"Created dataset with {len(self.all_slices)} 2D slices after filtering")
            
            # Visualize a few sample slices
            if self.visualize_samples and len(self.all_slices) > 0:
                self._visualize_sample_slices(5)  # Show 5 samples
        except Exception as e:
            print(f"Error preprocessing dataset: {e}")
            import traceback
            traceback.print_exc()
        
        # Debug: Check the first few slices
        if len(self.all_slices) > 0:
            print(f"===== DATASET DEBUG INFO =====")
            for i in range(min(3, len(self.all_slices))):
                img, lbl, idx = self.all_slices[i]
                print(f"Slice {i} - Image shape: {img.shape}, Label shape: {lbl.shape}, Slice index: {idx}")
                # Print label distribution
                unique_labels = torch.unique(lbl)
                print(f"  - Unique labels: {unique_labels}")
                for label_val in unique_labels:
                    count = (lbl == label_val).sum().item()
                    percentage = 100 * count / lbl.numel()
                    print(f"  - Label {label_val.item()}: {count} pixels ({percentage:.4f}%)")
            print("================================")
    
    def _resize_slice(self, slice_tensor, target_size):
        """Resize a 2D slice to the target size."""
        # Get the original size
        orig_size = slice_tensor.shape[1:]
        channels = slice_tensor.shape[0]
        
        # If already the right size, return
        if orig_size[0] == target_size[0] and orig_size[1] == target_size[1]:
            return slice_tensor
        
        # Do a simple resize (interpolate for image, nearest neighbor for mask)
        resized = torch.zeros((channels, target_size[0], target_size[1]), dtype=slice_tensor.dtype)
        
        # Simple center crop or pad
        if orig_size[0] > target_size[0] or orig_size[1] > target_size[1]:
            # Crop (take center)
            start_x = (orig_size[0] - target_size[0]) // 2
            start_y = (orig_size[1] - target_size[1]) // 2
            resized = slice_tensor[:, start_x:start_x+target_size[0], start_y:start_y+target_size[1]]
        else:
            # Pad (add borders)
            start_x = (target_size[0] - orig_size[0]) // 2
            start_y = (target_size[1] - orig_size[1]) // 2
            resized[:, start_x:start_x+orig_size[0], start_y:start_y+orig_size[1]] = slice_tensor
        
        return resized
    
    def _visualize_sample_slices(self, num_samples=5):
        """Visualize sample slices to help with debugging."""
        try:
            # Get a few random samples to visualize
            indices = np.random.choice(len(self.all_slices), min(num_samples, len(self.all_slices)), replace=False)
            
            plt.figure(figsize=(15, 3 * len(indices)))
            plt.suptitle("Sample 2D Slices (Image and Segmentation Overlay)", fontsize=16)
            
            for i, idx in enumerate(indices):
                img, lbl, slice_idx = self.all_slices[idx]
                
                # Extract the image (use the first modality for visualization)
                img_display = img[0].numpy()
                
                # Create RGB segmentation overlay
                lbl_overlay = np.zeros((*img_display.shape, 3))
                
                # Extract segmentation masks for different tumor regions
                # Create one-hot encoding
                background = (lbl[0] == 0).float().numpy()
                necrotic = (lbl[0] == 1).float().numpy()
                edema = (lbl[0] == 2).float().numpy()
                enhancing = (lbl[0] == 4).float().numpy()
                
                # Add colors: red=enhancing, green=edema, blue=necrotic
                lbl_overlay[..., 0] = enhancing * 1.0  # Red for enhancing
                lbl_overlay[..., 1] = edema * 1.0      # Green for edema
                lbl_overlay[..., 2] = necrotic * 1.0   # Blue for necrotic
                
                # Create two subplots side by side
                plt.subplot(len(indices), 2, i*2 + 1)
                plt.imshow(img_display, cmap='gray')
                plt.title(f"Image (Slice {slice_idx})")
                plt.axis('off')
                
                plt.subplot(len(indices), 2, i*2 + 2)
                plt.imshow(img_display, cmap='gray')
                plt.imshow(lbl_overlay, alpha=0.5)
                plt.title(f"Segmentation Overlay")
                plt.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig('sample_slices.png')
            print(f"Sample slices visualization saved to 'sample_slices.png'")
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing sample slices: {e}")
    
    def __len__(self):
        return len(self.all_slices)
    
    def __getitem__(self, idx):
        try:
            # Get pre-extracted 2D slice
            image, label, slice_idx = self.all_slices[idx]
            
            # Make sure we have proper tensor types
            if not isinstance(image, torch.Tensor):
                print(f"Warning: Image is not a tensor at index {idx}")
                image = torch.zeros((4, *self.target_size), dtype=torch.float32)
            
            if not isinstance(label, torch.Tensor):
                print(f"Warning: Label is not a tensor at index {idx}")
                label = torch.zeros((1, *self.target_size), dtype=torch.float32)
            
            # Debug: Check for enhancing tumor in this slice (label 3)
            has_et = torch.any(label == 3).item()
            if has_et and idx < 10:  # Only print for first few slices
                et_pixels = torch.sum(label == 3).item()
                total_pixels = label.numel()
                print(f"Slice {idx} contains enhancing tumor (label 3): {et_pixels} pixels ({100 * et_pixels / total_pixels:.4f}%)")
            
            # Create one-hot encoding for segmentation
            # Convert label to one-hot encoding: background (0), necrotic (1), edema (2), enhancing (3)
            one_hot = torch.zeros((4, *label.shape[1:]), dtype=torch.float32)
            
            # Extract specific classes - IMPORTANT: BraTS uses labels 0,1,2,3 (not 4)
            one_hot[0] = (label[0] == 0).float()  # Background
            one_hot[1] = (label[0] == 1).float()  # Necrotic and non-enhancing tumor
            one_hot[2] = (label[0] == 2).float()  # Peritumoral edema
            one_hot[3] = (label[0] == 3).float()  # GD-enhancing tumor
            
            # Debug: Check if one-hot encoding preserved enhancing tumor
            if has_et and idx < 10:
                et_pixels_onehot = torch.sum(one_hot[3]).item()
                print(f"  - After one-hot encoding: {et_pixels_onehot} pixels")
            
            # Make sure sizes are consistent - resize if needed
            if image.shape[1:] != one_hot.shape[1:]:
                # Resize by center cropping or padding to match target size
                target_size = self.target_size
                
                # For image
                if image.shape[1] > target_size[0] or image.shape[2] > target_size[1]:
                    # Center crop
                    start_x = (image.shape[1] - target_size[0]) // 2
                    start_y = (image.shape[2] - target_size[1]) // 2
                    image = image[:, start_x:start_x+target_size[0], start_y:start_y+target_size[1]]
                elif image.shape[1] < target_size[0] or image.shape[2] < target_size[1]:
                    # Pad
                    new_image = torch.zeros((image.shape[0], *target_size), dtype=image.dtype)
                    start_x = (target_size[0] - image.shape[1]) // 2
                    start_y = (target_size[1] - image.shape[2]) // 2
                    new_image[:, start_x:start_x+image.shape[1], start_y:start_y+image.shape[2]] = image
                    image = new_image
                
                # For one_hot
                if one_hot.shape[1] > target_size[0] or one_hot.shape[2] > target_size[1]:
                    # Center crop
                    start_x = (one_hot.shape[1] - target_size[0]) // 2
                    start_y = (one_hot.shape[2] - target_size[1]) // 2
                    one_hot = one_hot[:, start_x:start_x+target_size[0], start_y:start_y+target_size[1]]
                elif one_hot.shape[1] < target_size[0] or one_hot.shape[2] < target_size[1]:
                    # Pad
                    new_one_hot = torch.zeros((one_hot.shape[0], *target_size), dtype=one_hot.dtype)
                    start_x = (target_size[0] - one_hot.shape[1]) // 2
                    start_y = (target_size[1] - one_hot.shape[2]) // 2
                    new_one_hot[:, start_x:start_x+one_hot.shape[1], start_y:start_y+one_hot.shape[2]] = one_hot
                    one_hot = new_one_hot
            
            # Verify one-hot encoding is valid (sums to 1 across channels)
            sum_channels = one_hot.sum(dim=0)
            if not torch.all(sum_channels <= 1.0001) or not torch.all(sum_channels >= 0.9999):
                print(f"Warning: One-hot encoding is invalid at index {idx}. Sum across channels: min={sum_channels.min().item()}, max={sum_channels.max().item()}")
                # Fix one-hot encoding
                max_indices = torch.argmax(one_hot, dim=0)
                fixed_one_hot = torch.zeros_like(one_hot)
                for c in range(one_hot.shape[0]):
                    fixed_one_hot[c] = (max_indices == c).float()
                one_hot = fixed_one_hot
            
            return image, one_hot, slice_idx
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            # Return a default tensor in case of error
            dummy_image = torch.zeros((4, *self.target_size), dtype=torch.float32)
            dummy_label = torch.zeros((4, *self.target_size), dtype=torch.float32)
            dummy_label[0] = 1.0  # All background
            return dummy_image, dummy_label, 0

# Function to calculate Dice score
def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

# Function to calculate Hausdorff distance using MONAI's implementation
def hausdorff_distance(pred, target):
    """
    Compute the 95% Hausdorff Distance using medpy implementation.
    
    Args:
        pred (torch.Tensor): Prediction tensor (binary)
        target (torch.Tensor): Target tensor (binary)
    
    Returns:
        float: 95% Hausdorff distance
    """
    # Convert tensors to numpy arrays
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)
    
    # Check if either array is empty (no positive pixels)
    if not np.any(pred_np) or not np.any(target_np):
        return float('nan')
    
    try:
        return hd95(pred_np, target_np)
    except Exception as e:
        print(f"Error computing Hausdorff distance: {e}")
        return float('nan')

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_dir='checkpoints'):
    """
    Train the model.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        num_epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        model: The trained model
        history: Training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_hausdorff': []
    }
    
    # Initialize best validation metrics
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in pbar:
                # Get data
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                # Check for NaN values
                if torch.isnan(images).any() or torch.isnan(labels).any():
                    print("Warning: NaN values detected in input data. Skipping batch.")
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("Warning: NaN loss detected. Skipping batch.")
                    continue
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        
        # Update history
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_hausdorff'].append(val_metrics['hausdorff'])
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}')
        print(f'  Val Dice: {val_metrics["dice"]:.4f}')
        print(f'  Val Hausdorff: {val_metrics["hausdorff"]:.4f}')
        
        # Save checkpoint if validation loss improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            print(f'  Checkpoint saved to {checkpoint_path}')
    
    return model, history

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
def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        device: Device to run evaluation on
        criterion: Loss function
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    dice_scores = []
    hausdorff_distances = []
    
    # Counters for debugging
    total_samples = 0
    empty_gt_samples = 0
    empty_pred_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert outputs to binary predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for each class (excluding background)
            for c in range(1, 4):  # Classes 1, 2, 3 (excluding background)
                for i in range(preds.shape[0]):  # For each sample in batch
                    pred_c = (preds[i] == c).float()
                    label_c = (labels[i, c] == 1).float()
                    
                    # Count samples
                    total_samples += 1
                    if not torch.any(label_c):
                        empty_gt_samples += 1
                        continue
                    
                    if not torch.any(pred_c):
                        empty_pred_samples += 1
                        continue
                    
                    # Calculate Dice score
                    dice = dice_coefficient(pred_c, label_c)
                    if not torch.isnan(dice):
                        dice_scores.append(dice.item())
                    
                    # Calculate Hausdorff distance
                    hd = hausdorff_distance(pred_c, label_c)
                    if not np.isnan(hd):
                        hausdorff_distances.append(hd)
    
    # Print debugging information
    print(f"Total samples evaluated: {total_samples}")
    print(f"Samples with empty ground truth: {empty_gt_samples}")
    print(f"Samples with empty predictions: {empty_pred_samples}")
    print(f"Valid dice scores: {len(dice_scores)}")
    print(f"Valid Hausdorff distances: {len(hausdorff_distances)}")
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_hd = np.mean(hausdorff_distances) if hausdorff_distances else float('inf')
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'hausdorff': avg_hd
    }

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
    if not examples:
        print("No examples to visualize")
        return
        
    try:
        plt.figure(figsize=(15, 10))
        
        for i, example_data in enumerate(examples[:5]):  # Show up to 5 examples
            if i >= 5:  # Limit to 5 examples
                break
                
            if len(example_data) < 3:
                print(f"Warning: Example {i} does not contain enough data elements")
                continue
                
            # Unpack the example data
            images, labels, preds = example_data[:3]
            slice_idx = example_data[3] if len(example_data) > 3 else torch.zeros(1)
            
            # Make sure we have data in the batch
            if images.size(0) == 0 or labels.size(0) == 0 or preds.size(0) == 0:
                print(f"Warning: Example {i} contains empty tensors")
                continue
                
            # Get the first image in the batch
            image = images[0]  # Get first image from batch
            label = labels[0]  # Get first label from batch
            pred = preds[0]    # Get first prediction from batch
            
            # Check dimensions and channels
            if image.dim() < 3 or label.dim() < 3:
                print(f"Warning: Example {i} has incorrect dimensions - image: {image.shape}, label: {label.shape}")
                continue
                
            # Extract T1ce modality for visualization (or use first channel if not clear)
            t1ce = image[0].numpy() if image.size(0) > 0 else np.zeros((128, 128))
            
            try:
                # Convert to binary masks
                # Labels are already one-hot encoded
                # Channel 0: background (label 0)
                # Channel 1: necrotic and non-enhancing tumor (label 1)
                # Channel 2: peritumoral edema (label 2)
                # Channel 3: enhancing tumor (label 4)
                gt_et = label[3].float().numpy() if label.size(0) > 3 else np.zeros_like(t1ce)  # Enhancing tumor (label 4)
                
                # Use logical operations for combining masks
                gt_tc = ((label[1] > 0.5) | (label[3] > 0.5)).float().numpy() if label.size(0) > 3 else np.zeros_like(t1ce)
                gt_wt = ((label[1] > 0.5) | (label[2] > 0.5) | (label[3] > 0.5)).float().numpy() if label.size(0) > 3 else np.zeros_like(t1ce)
                
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
                plt.title(f'Ground Truth (Slice {slice_idx[0].item() if isinstance(slice_idx, torch.Tensor) else 0})')
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
            except Exception as e:
                print(f"Error visualizing example {i}: {e}")
                continue
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', f'segmentation_results_fold{fold+1}.png'))
        plt.close()
    except Exception as e:
        print(f"Error in visualization: {e}")

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
    
    try:
        for batch in tqdm(data_loader, desc='Analyzing dataset'):
            # Unpack batch - handle both list and tuple formats
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                # Skip this batch if format is unexpected
                continue
                
            total_slices += images.size(0)
            
            for i in range(images.size(0)):
                # Extract enhancing tumor mask (channel 3 for class 4)
                if labels.size(1) > 3:  # Make sure we have enough channels
                    et_mask = labels[i, 3]  # Already a float tensor
                    
                    # Count pixels
                    et_pixels = torch.sum(et_mask).item()
                    total_pixels = et_mask.numel()
                    
                    if et_pixels > 0:
                        slices_with_et += 1
                        et_pixel_counts.append(et_pixels)
                        et_pixel_percentages.append(100 * et_pixels / total_pixels)
    except Exception as e:
        print(f"Warning during dataset analysis: {e}")
        # Continue with partial results if any
    
    # Print statistics
    print(f"Total slices analyzed: {total_slices}")
    print(f"Slices containing enhancing tumor: {slices_with_et} ({100 * slices_with_et / max(total_slices, 1):.2f}%)")
    
    if slices_with_et > 0:
        print(f"Average ET pixels per slice (when present): {np.mean(et_pixel_counts):.2f}")
        print(f"Median ET pixels per slice (when present): {np.median(et_pixel_counts):.2f}")
        print(f"Min ET pixels per slice: {np.min(et_pixel_counts):.2f}")
        print(f"Max ET pixels per slice: {np.max(et_pixel_counts):.2f}")
        print(f"Average percentage of slice covered by ET: {np.mean(et_pixel_percentages):.4f}%")
    
    print("==========================================================")
    
    return slices_with_et / max(total_slices, 1)  # Avoid division by zero

# Main function
def main():
    """
    Main function to run the brain tumor segmentation pipeline.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set hyperparameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.0001
    target_size = (128, 128)
    
    # Set paths
    data_dir = 'data'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = BrainTumorDataset(
        data_dir=data_dir,
        transform=True,
        train=True,
        resize=True,
        target_size=target_size,
        visualize_samples=True
    )
    
    val_dataset = BrainTumorDataset(
        data_dir=data_dir,
        transform=False,
        train=False,
        resize=True,
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Check if data loaders have data
    if len(train_loader) == 0:
        print("Warning: Training data loader is empty.")
        return
    
    if len(val_loader) == 0:
        print("Warning: Validation data loader is empty.")
        return
    
    # Create model
    print("Creating model...")
    model = UNet(in_channels=4, out_channels=4).to(device)
    
    # Define loss function and optimizer
    criterion = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train or evaluate
    print("Do you want to train or evaluate the model? (train/eval)")
    choice = input().lower()
    
    if choice == 'train':
        # Train the model
        print("Training model...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_dice'], label='Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        
        plt.subplot(1, 3, 3)
        plt.plot(history['val_hausdorff'], label='HD95')
        plt.title('Hausdorff Distance')
        plt.xlabel('Epoch')
        plt.ylabel('HD95')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
    elif choice == 'eval':
        # Load model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_10.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
            
            # Evaluate model
            print("Evaluating model...")
            metrics = evaluate_model(model, val_loader, device, criterion)
            
            print("\nEvaluation Results:")
            print(f"Dice Score: {metrics['dice']:.4f}")
            print(f"Hausdorff Distance: {metrics['hausdorff']:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    else:
        print("Invalid choice. Please enter 'train' or 'eval'.")

def dice_coefficient(pred, target):
    """
    Compute the Dice coefficient between two binary tensors.
    
    Args:
        pred (torch.Tensor): Prediction tensor (binary)
        target (torch.Tensor): Target tensor (binary)
    
    Returns:
        torch.Tensor: Dice coefficient
    """
    smooth = 1e-5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    
    if union < smooth:
        return torch.tensor(0.0, device=pred.device)
    
    return (2.0 * intersection + smooth) / (union + smooth)

if __name__ == "__main__":
    main()

