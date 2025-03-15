# Deep Learning-based Brain Tumor Segmentation Using MRI
## CAP 5516 - Medical Image Computing (Spring 2025)

This repository contains a modular implementation of a brain tumor segmentation model using PyTorch and MONAI.

## Project Structure

The code has been organized into the following modules:

- `models.py`: Contains the model architecture definitions (UNet)
- `dataset.py`: Contains the dataset class and data loading functions
- `utils.py`: Contains utility functions for metrics, visualization, etc.
- `train.py`: Contains functions for training and evaluating the model
- `main.py`: Main script to run the entire pipeline

## Implementation Details

### System Specifications
- Processor: Intel(R) Core(TM) i9-14900K 3.20 GHz
- RAM: 64.0 GB (63.8 GB usable)
- System Type: 64-bit operating system, x64-based processor
- GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)

### Network Architecture
- **Model**: 2D U-Net
- **Approach**: Process MRI slice by slice (2D)
- **Input**: 4 MRI modalities (channels)
- **Output**: 4 classes (background, necrotic/non-enhancing tumor, peritumoral edema, enhancing tumor)
- **Encoder**: 4 blocks with double convolution and max pooling
- **Bottleneck**: Double convolution
- **Decoder**: 4 blocks with transposed convolution and skip connections
- **Final layer**: 1x1 convolution to map to output classes

### Training Parameters
- Learning rate: 0.0001
- Batch size: 8 (memory-efficient)
- Training epochs: 10
- Optimizer: Adam
- Loss function: DiceCELoss (MONAI implementation)
- Gradient clipping: 1.0 (for training stability)

### Dataset Statistics
- Dataset: BraTS Brain Tumor Segmentation Challenge
- Segmentation classes:
  - Background (label 0)
  - Necrotic and non-enhancing tumor (label 1)
  - Peritumoral edema (label 2)
  - GD-enhancing tumor (label 3)

## How to Run

1. Install the required dependencies:
```
pip install torch monai nibabel matplotlib tqdm medpy
```

2. Run the main script:
```
python main.py
```

3. Choose a mode:
   - `train`: Train a new model
   - `eval`: Evaluate an existing model
   - `test`: Run tests to verify metrics implementation

## Modules Explanation

### models.py

This module contains the model architecture definitions:
- `DoubleConv`: A double convolution block used in UNet
- `UNet`: The UNet model architecture for segmentation
- `init_weights`: A function to initialize model weights for better training stability

### dataset.py

This module handles data loading and preprocessing:
- `BrainTumorDataset`: A PyTorch Dataset class for loading and preprocessing brain tumor data
  - Extracts 2D slices from 3D volumes
  - Applies resizing for memory efficiency
  - Creates one-hot encoded segmentation masks
  - Visualizes sample slices for debugging
- `create_data_loaders`: A function to create DataLoader objects for training and validation

### utils.py

This module contains utility functions:
- `dice_coefficient`: Function to calculate the Dice coefficient
- `hausdorff_distance`: Function to calculate the Hausdorff distance using medpy
- `plot_losses`: Function to plot training and validation losses
- `visualize_results`: Function to visualize segmentation results
- `set_seed`: Function to set random seeds for reproducibility
- `test_metrics`: Function to test metric implementations with simple cases

### train.py

This module contains functions for training and evaluation:
- `train_model`: Function to train the model
  - Handles gradient clipping for stability
  - Tracks training and validation metrics
  - Saves model checkpoints
- `evaluate_model`: Function to evaluate the model on validation data
  - Calculates class-specific metrics
  - Collects examples for visualization

### main.py

This is the main script that ties everything together:
- Sets up the environment (device, random seed)
- Creates datasets and data loaders
- Initializes the model, loss function, and optimizer
- Handles training, evaluation, or testing based on user input

## Memory Efficiency Features

- **Slice-based processing**: Processes 3D volumes slice by slice to reduce memory usage
- **Resizing option**: Resizes slices to a smaller target size (e.g., 128x128)
- **On-demand loading**: Loads and processes data as needed
- **Gradient clipping**: Prevents exploding gradients
- **Error handling**: Checks for NaN values and handles memory-intensive operations

## Evaluation Metrics

The model is evaluated using:
- **Dice coefficient**: Measures overlap between prediction and ground truth
- **Hausdorff distance (95%)**: Measures the distance between prediction and ground truth boundaries

Metrics are calculated for:
- Each individual class (Background, Necrotic, Edema, Enhancing)
- Overall performance across all classes

## Visualization

The code includes visualization tools for:
- Sample slices from the dataset with segmentation overlays
- Model predictions compared to ground truth
- Training and validation metrics over time

## Debugging and Testing

- **Test mode**: Verifies metric implementations with simple test cases
- **Visualization tools**: Helps identify issues in data processing and model predictions
- **Detailed logging**: Provides information about the training and evaluation process
- **Error handling**: Catches and reports issues during execution

## Dataset Structure

The dataset should be organized in the following structure:
```
Task01_BrainTumour/
├── imagesTr/
│   ├── BRATS_XXX.nii.gz
│   └── ...
└── labelsTr/
    ├── BRATS_XXX.nii.gz
    └── ...
```

## Benefits of Modular Structure

1. **Improved Readability**: Each module has a clear purpose, making the code easier to understand.
2. **Better Maintainability**: Changes to one part of the code (e.g., model architecture) don't require changes to other parts.
3. **Easier Debugging**: Issues can be isolated to specific modules, making debugging more straightforward.
4. **Code Reusability**: Functions and classes can be reused in other projects or experiments.
5. **Collaboration**: Multiple people can work on different modules simultaneously without conflicts. 