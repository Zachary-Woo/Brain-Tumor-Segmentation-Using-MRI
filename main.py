import os
import torch
import torch.optim as optim
from monai.losses import DiceCELoss

from models import UNet, init_weights
from dataset import BrainTumorDataset, create_data_loaders
from train import train_model, evaluate_model
from utils import plot_losses, visualize_results, set_seed, test_metrics

def main():
    """
    Main function to run the brain tumor segmentation pipeline.
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set hyperparameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.0001
    target_size = (128, 128)
    
    # Set paths
    data_dir = 'Task01_BrainTumour'
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Ask user for mode
    print("Choose a mode: (train/eval/test)")
    choice = input().lower()
    
    # Test mode - verify metrics are working
    if choice == 'test':
        print("Running in test mode to verify metrics...")
        test_metrics()
        return
    
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
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        batch_size=batch_size
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
    model = UNet(in_channels=4, out_channels=4)
    model = init_weights(model)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        plot_losses(history, save_path=os.path.join(results_dir, 'training_history.png'))
        
    elif choice == 'eval':
        # Load model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_10.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
            
            # Evaluate model
            print("Evaluating model...")
            val_loss, val_dice, metrics = evaluate_model(model, val_loader, device, criterion)
            
            print("\nEvaluation Results:")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Overall Dice Score: {val_dice:.4f}")
            
            # Print class-specific metrics
            for c in range(4):
                class_name = ["Background", "Necrotic", "Edema", "Enhancing"][c]
                print(f"{class_name} - Dice: {metrics['class_dice'][c]:.4f}, Hausdorff: {metrics['class_hausdorff'][c]:.4f}")
            
            # Visualize results
            visualize_results(metrics['examples'], output_dir=results_dir)
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    else:
        print("Invalid choice. Please enter 'train', 'eval', or 'test'.")

if __name__ == "__main__":
    main() 