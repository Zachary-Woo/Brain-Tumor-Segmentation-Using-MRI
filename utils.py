import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95, dc, assd, precision
import random
import logging

def dice_coefficient(pred, target):
    """
    Compute Dice coefficient between two binary tensors.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        
    Returns:
        float: Dice coefficient
    """
    smooth = 1e-5
    
    # Convert tensors to numpy arrays if they are not already
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def hausdorff_distance(pred, target):
    """
    Compute 95% Hausdorff Distance between two binary tensors using medpy.
    
    Args:
        pred (torch.Tensor): Prediction tensor
        target (torch.Tensor): Target tensor
        
    Returns:
        float: 95% Hausdorff Distance
    """
    # Convert tensors to numpy arrays if they are not already
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Check if either array is empty (no positive pixels)
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return float('inf')
    
    try:
        return hd95(pred, target)
    except Exception as e:
        print(f"Error computing Hausdorff distance: {e}")
        return float('inf')

def plot_losses(history, save_path='training_history.png'):
    """
    Plot training and validation losses and metrics.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def visualize_results(examples, output_dir='results'):
    """
    Visualize model predictions compared to ground truth.
    
    Args:
        examples (list): List of dictionaries containing image, ground truth, and prediction
        output_dir (str, optional): Directory to save visualizations. Defaults to 'results'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not examples:
        print("No examples to visualize")
        return
    
    for i, example in enumerate(examples):
        image = example['image'].squeeze().cpu().numpy()  # (C, H, W) -> (C, H, W)
        ground_truth = example['ground_truth'].squeeze().cpu().numpy()  # (1, H, W) -> (H, W)
        prediction = example['prediction'].squeeze().cpu().numpy()  # (1, H, W) -> (H, W)
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image (first channel - T1)
        axes[0].imshow(image[0], cmap='gray')
        axes[0].set_title('Original Image (T1)')
        axes[0].axis('off')
        
        # Create color maps for segmentation
        # Background: black, Necrotic: red, Edema: green, Enhancing: blue
        cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green', 'blue'])
        
        # Plot ground truth
        axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=3)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=3)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{i}.png'))
        plt.close()
    
    print(f"Saved {len(examples)} visualization images to {output_dir}")

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_metrics():
    """
    Test function to verify that metrics are working correctly.
    Creates simple test cases and computes metrics.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MetricsTest")
    
    # Test case 1: Perfect prediction
    logger.info("Test Case 1: Perfect prediction")
    pred1 = np.zeros((10, 10))
    pred1[2:8, 2:8] = 1
    target1 = np.zeros((10, 10))
    target1[2:8, 2:8] = 1
    
    # Convert NumPy arrays to PyTorch tensors
    pred1_tensor = torch.tensor(pred1, dtype=torch.float32)
    target1_tensor = torch.tensor(target1, dtype=torch.float32)
    
    dice1 = dice_coefficient(pred1_tensor, target1_tensor)
    medpy_dice1 = dc(pred1, target1)
    hd1 = hausdorff_distance(pred1_tensor, target1_tensor)
    
    logger.info(f"Dice coefficient: {dice1:.4f} (Our implementation)")
    logger.info(f"Dice coefficient: {medpy_dice1:.4f} (medpy implementation)")
    logger.info(f"Hausdorff distance: {hd1:.4f}")
    
    # Test case 2: Partial overlap
    logger.info("\nTest Case 2: Partial overlap")
    pred2 = np.zeros((10, 10))
    pred2[2:8, 2:8] = 1
    target2 = np.zeros((10, 10))
    target2[4:10, 4:10] = 1
    
    # Convert NumPy arrays to PyTorch tensors
    pred2_tensor = torch.tensor(pred2, dtype=torch.float32)
    target2_tensor = torch.tensor(target2, dtype=torch.float32)
    
    dice2 = dice_coefficient(pred2_tensor, target2_tensor)
    medpy_dice2 = dc(pred2, target2)
    hd2 = hausdorff_distance(pred2_tensor, target2_tensor)
    
    logger.info(f"Dice coefficient: {dice2:.4f} (Our implementation)")
    logger.info(f"Dice coefficient: {medpy_dice2:.4f} (medpy implementation)")
    logger.info(f"Hausdorff distance: {hd2:.4f}")
    
    # Test case 3: No overlap
    logger.info("\nTest Case 3: No overlap")
    pred3 = np.zeros((10, 10))
    pred3[0:5, 0:5] = 1
    target3 = np.zeros((10, 10))
    target3[5:10, 5:10] = 1
    
    # Convert NumPy arrays to PyTorch tensors
    pred3_tensor = torch.tensor(pred3, dtype=torch.float32)
    target3_tensor = torch.tensor(target3, dtype=torch.float32)
    
    dice3 = dice_coefficient(pred3_tensor, target3_tensor)
    medpy_dice3 = dc(pred3, target3)
    hd3 = hausdorff_distance(pred3_tensor, target3_tensor)
    
    logger.info(f"Dice coefficient: {dice3:.4f} (Our implementation)")
    logger.info(f"Dice coefficient: {medpy_dice3:.4f} (medpy implementation)")
    logger.info(f"Hausdorff distance: {hd3}")
    
    # Test case 4: Empty prediction
    logger.info("\nTest Case 4: Empty prediction")
    pred4 = np.zeros((10, 10))
    target4 = np.zeros((10, 10))
    target4[2:8, 2:8] = 1
    
    # Convert NumPy arrays to PyTorch tensors
    pred4_tensor = torch.tensor(pred4, dtype=torch.float32)
    target4_tensor = torch.tensor(target4, dtype=torch.float32)
    
    dice4 = dice_coefficient(pred4_tensor, target4_tensor)
    medpy_dice4 = dc(pred4, target4)
    hd4 = hausdorff_distance(pred4_tensor, target4_tensor)
    
    logger.info(f"Dice coefficient: {dice4:.4f} (Our implementation)")
    logger.info(f"Dice coefficient: {medpy_dice4:.4f} (medpy implementation)")
    logger.info(f"Hausdorff distance: {hd4}")
    
    # Test case 5: Empty target
    logger.info("\nTest Case 5: Empty target")
    pred5 = np.zeros((10, 10))
    pred5[2:8, 2:8] = 1
    target5 = np.zeros((10, 10))
    
    # Convert NumPy arrays to PyTorch tensors
    pred5_tensor = torch.tensor(pred5, dtype=torch.float32)
    target5_tensor = torch.tensor(target5, dtype=torch.float32)
    
    dice5 = dice_coefficient(pred5_tensor, target5_tensor)
    medpy_dice5 = dc(pred5, target5)
    hd5 = hausdorff_distance(pred5_tensor, target5_tensor)
    
    logger.info(f"Dice coefficient: {dice5:.4f} (Our implementation)")
    logger.info(f"Dice coefficient: {medpy_dice5:.4f} (medpy implementation)")
    logger.info(f"Hausdorff distance: {hd5}")
    
    # Test case 6: Both empty
    logger.info("\nTest Case 6: Both empty")
    pred6 = np.zeros((10, 10))
    target6 = np.zeros((10, 10))
    
    # Convert NumPy arrays to PyTorch tensors
    pred6_tensor = torch.tensor(pred6, dtype=torch.float32)
    target6_tensor = torch.tensor(target6, dtype=torch.float32)
    
    dice6 = dice_coefficient(pred6_tensor, target6_tensor)
    # medpy_dice6 = dc(pred6, target6)  # This would raise an error in medpy
    
    logger.info(f"Dice coefficient: {dice6:.4f} (Our implementation)")
    logger.info("medpy implementation would raise an error for empty arrays")
    
    # Summary
    logger.info("\nSummary of test cases:")
    logger.info(f"Test Case 1 (Perfect): Dice={dice1:.4f}, HD={hd1:.4f}")
    logger.info(f"Test Case 2 (Partial): Dice={dice2:.4f}, HD={hd2:.4f}")
    logger.info(f"Test Case 3 (No overlap): Dice={dice3:.4f}, HD={hd3}")
    logger.info(f"Test Case 4 (Empty pred): Dice={dice4:.4f}, HD={hd4}")
    logger.info(f"Test Case 5 (Empty target): Dice={dice5:.4f}, HD={hd5}")
    logger.info(f"Test Case 6 (Both empty): Dice={dice6:.4f}")
    
    # Visual verification
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(pred1, cmap='Blues')
    plt.title('Test Case 1: Prediction')
    plt.subplot(2, 3, 4)
    plt.imshow(target1, cmap='Reds')
    plt.title('Test Case 1: Target')
    
    plt.subplot(2, 3, 2)
    plt.imshow(pred2, cmap='Blues')
    plt.title('Test Case 2: Prediction')
    plt.subplot(2, 3, 5)
    plt.imshow(target2, cmap='Reds')
    plt.title('Test Case 2: Target')
    
    plt.subplot(2, 3, 3)
    plt.imshow(pred3, cmap='Blues')
    plt.title('Test Case 3: Prediction')
    plt.subplot(2, 3, 6)
    plt.imshow(target3, cmap='Reds')
    plt.title('Test Case 3: Target')
    
    plt.tight_layout()
    plt.savefig('metrics_test_visualization.png')
    logger.info("Saved visualization of test cases to metrics_test_visualization.png")
    plt.close()
    return True 