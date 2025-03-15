# Brain Tumor Segmentation - Modular Structure

This repository contains a modular implementation of a brain tumor segmentation model using PyTorch and MONAI.

## Project Structure

The code has been organized into the following modules:

- `models.py`: Contains the model architecture definitions (UNet)
- `dataset.py`: Contains the dataset class and data loading functions
- `utils.py`: Contains utility functions for metrics, visualization, etc.
- `train.py`: Contains functions for training and evaluating the model
- `main.py`: Main script to run the entire pipeline

## How to Run

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the main script:
```
python main.py
```

3. Choose whether to train a new model or evaluate an existing one.

## Modules Explanation

### models.py

This module contains the model architecture definitions:
- `DoubleConv`: A double convolution block used in UNet
- `UNet`: The UNet model architecture for segmentation
- `init_weights`: A function to initialize model weights for better training stability

### dataset.py

This module handles data loading and preprocessing:
- `BrainTumorDataset`: A PyTorch Dataset class for loading and preprocessing brain tumor data
- `create_data_loaders`: A function to create DataLoader objects for training and validation

### utils.py

This module contains utility functions:
- `dice_coefficient`: Function to calculate the Dice coefficient
- `hausdorff_distance`: Function to calculate the Hausdorff distance using medpy
- `plot_losses`: Function to plot training and validation losses
- `visualize_results`: Function to visualize segmentation results
- `set_seed`: Function to set random seeds for reproducibility

### train.py

This module contains functions for training and evaluation:
- `train_model`: Function to train the model
- `evaluate_model`: Function to evaluate the model on validation data

### main.py

This is the main script that ties everything together:
- Sets up the environment (device, random seed)
- Creates datasets and data loaders
- Initializes the model, loss function, and optimizer
- Handles training or evaluation based on user input

## Benefits of Modular Structure

1. **Improved Readability**: Each module has a clear purpose, making the code easier to understand.
2. **Better Maintainability**: Changes to one part of the code (e.g., model architecture) don't require changes to other parts.
3. **Easier Debugging**: Issues can be isolated to specific modules, making debugging more straightforward.
4. **Code Reusability**: Functions and classes can be reused in other projects or experiments.
5. **Collaboration**: Multiple people can work on different modules simultaneously without conflicts.

## Debugging Tips

1. **Model Issues**: Check `models.py` for architecture problems or weight initialization.
2. **Data Issues**: Check `dataset.py` for data loading, preprocessing, or augmentation problems.
3. **Training Issues**: Check `train.py` for issues with the training loop, loss calculation, or optimization.
4. **Metric Issues**: Check `utils.py` for problems with metric calculations.
5. **General Issues**: Check `main.py` for configuration problems or integration issues.

Each module has detailed logging and error handling to help identify issues during execution. 