import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd
)
from monai.data import Dataset as MonaiDataset

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=False, train=True, resize=True, target_size=(128, 128), visualize_samples=False):
        """
        Brain tumor segmentation dataset.
        
        Args:
            data_dir (str): Path to the data directory
            transform (bool): Whether to apply data augmentation
            train (bool): Whether this is a training dataset
            resize (bool): Whether to resize images
            target_size (tuple): Target size for resizing
            visualize_samples (bool): Whether to visualize sample slices
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.resize = resize
        self.target_size = target_size
        self.visualize_samples = visualize_samples
        
        # Define paths
        images_dir = os.path.join(data_dir, "imagesTr")
        labels_dir = os.path.join(data_dir, "labelsTr")
        
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
        self.image_files = [os.path.join(images_dir, f) for f in valid_files]
        self.label_files = [os.path.join(labels_dir, f) for f in valid_files]
        
        # Define simplified transforms for loading and preprocessing 3D data
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
        
        # We'll apply simpler transforms before slicing the data to 2D
        self.transforms = Compose(base_transforms)
        
        # Create data list for MONAI dataset
        self.data_dicts = [
            {"image": image_file, "label": label_file}
            for image_file, label_file in zip(self.image_files, self.label_files)
        ]
        
        # Initialize MONAI dataset with simpler transforms
        self.dataset = MonaiDataset(self.data_dicts, transform=self.transforms)
        
        # Store slices for all volumes to create a flat dataset
        self.all_slices = []
        
        # Pre-process datasets to extract 2D slices from 3D volumes
        self._extract_2d_slices()
    
    def _extract_2d_slices(self):
        """Extract 2D slices from 3D volumes"""
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
                    if self.transform and not has_tumor:
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
                enhancing = (lbl[0] == 3).float().numpy()
                
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
            
            return image, one_hot
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            # Return a default tensor in case of error
            dummy_image = torch.zeros((4, *self.target_size), dtype=torch.float32)
            dummy_label = torch.zeros((4, *self.target_size), dtype=torch.float32)
            dummy_label[0] = 1.0  # All background
            return dummy_image, dummy_label

def create_data_loaders(train_dataset, val_dataset, batch_size=8, num_workers=0):
    """
    Create data loaders for training and validation datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader 