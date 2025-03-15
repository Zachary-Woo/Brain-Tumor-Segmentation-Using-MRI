# Deep Learning-based Brain Tumor Segmentation Using MRI
## CAP 5516 - Medical Image Computing (Spring 2025)
### Zachary Wood
### Programming Assignment #2

Note: Due to Canvas submission limitations allowing only one submission, the complete project report (Programming Assignment #2 Report.pdf) has been included in this GitHub repository alongside the code.

Disclaimer: The results in the report and shown below are from version 2. Version 3 is still running at time of the initial upload so the results are unclear in comparison.

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
- Learning rate: 0.001
- Batch size: 4 (memory-efficient)
- Training epochs: 10
- Optimizer: Adam
- Loss function: Dice Loss with class weights [0.1, 0.3, 0.3, 0.3]
- Learning rate scheduling: ReduceLROnPlateau
  - Mode: min (monitoring validation loss)
  - Factor: 0.5
  - Patience: 3 epochs

### Data Augmentation
- Random rotation (0°, 90°, 180°, 270°)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random brightness adjustment (factor: 0.8-1.2)

### Model-Specific Optimizations
- Memory-efficient data loading (loading slices on-demand)
- Slice-based processing to handle large 3D volumes
- One-hot encoding for segmentation masks
- Class weighting to handle class imbalance
- Error handling for memory-intensive operations

### Dataset Statistics
- Dataset: BraTS Brain Tumor Segmentation Challenge
- 5-fold cross-validation on training set
- Segmentation classes:
  - Background (label 0)
  - Necrotic and non-enhancing tumor (label 1)
  - Peritumoral edema (label 2)
  - GD-enhancing tumor (label 4, mapped to index 3)

## Results

### Segmentation Performance
- **Dice Scores**:
  - Enhancing Tumor (ET): 1.0000
  - Tumor Core (TC): 0.1586
  - Whole Tumor (WT): 0.2285
- **Hausdorff Distance (95%)**:
  - Enhancing Tumor (ET): 0.0000
  - Tumor Core (TC): 85.2865
  - Whole Tumor (WT): 78.2804

### Cross-Validation Results
- 5-fold cross-validation performed
- Detailed metrics for each fold:

| Fold | Dice ET | Dice TC | Dice WT | HD95 ET | HD95 TC | HD95 WT |
|------|---------|---------|---------|---------|---------|---------|
| 1    | 1.0000  | 0.2110  | 0.2615  | 0.0000  | 82.5726 | 81.9243 |
| 2    | 1.0000  | 0.1449  | 0.2320  | 0.0000  | 77.6252 | 74.1281 |
| 3    | 1.0000  | 0.1271  | 0.2490  | 0.0000  | 86.8297 | 62.5787 |
| 4    | 1.0000  | 0.2221  | 0.2782  | 0.0000  | 81.7443 | 74.9980 |
| 5    | 1.0000  | 0.0878  | 0.1220  | 0.0000  | 97.6608 | 97.7728 |
| **Avg** | **1.0000** | **0.1586** | **0.2285** | **0.0000** | **85.2865** | **78.2804** |

### Visualization
- Segmentation masks overlaid on MRI slices
- Color coding:
  - Red: Enhancing tumor (ET)
  - Blue: Tumor core (TC)
  - Green: Whole tumor (WT)
- Visualizations saved in `results_v2/visualizations/`

## Analysis

### Model Performance
- **Enhancing Tumor (ET)**:
  - Perfect Dice score of 1.0 and HD95 of 0.0 across all folds
  - This suggests potential issues with the evaluation or data processing
  - Further investigation needed to validate these results
  
- **Tumor Core (TC)**:
  - Lower Dice scores (average 0.1586)
  - High Hausdorff distances (average 85.2865)
  - Indicates challenges in accurately segmenting the tumor core
  
- **Whole Tumor (WT)**:
  - Moderate Dice scores (average 0.2285)
  - High Hausdorff distances (average 78.2804)
  - Better than TC but still shows room for improvement

### Memory Efficiency
- Implemented slice-by-slice processing to handle memory constraints
- On-demand data loading to reduce RAM usage
- Reduced batch size to accommodate GPU memory limitations
- Error handling for memory-intensive operations

### Challenges and Solutions
- **Memory Management**:
  - Challenge: Large 3D volumes causing memory errors
  - Solution: Slice-based processing and on-demand loading
  
- **Class Imbalance**:
  - Challenge: Enhancing tumor regions much smaller than background
  - Solution: Class weighting in Dice Loss function [0.1, 0.3, 0.3, 0.3]
  
- **3D Context**:
  - Challenge: 2D slices lose 3D context
  - Solution: Focus on slices with tumor presence

### Observations and Future Work
- The perfect ET scores require further investigation
- TC and WT segmentation performance could be improved
- Potential improvements:
  - 3D U-Net to capture volumetric context
  - More advanced data augmentation
  - Ensemble methods combining multiple models
  - Attention mechanisms to focus on tumor regions

## Running the Code

### Dataset Structure
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

### Requirements
- Python 3.8+
- PyTorch 1.8+
- nibabel
- numpy
- matplotlib
- pandas
- scikit-learn
- tqdm
- medpy (for Hausdorff distance calculation)

### Running the Training Script
```bash
python assignment2.py
```

### Options
- When prompted, choose `train` for training a new model or `pretrained` to use existing models
- Results will be saved in the `results_2d_unet` directory
- Visualizations will be saved in `results_2d_unet/visualizations`
- Model checkpoints will be saved in `results_2d_unet/checkpoints`
- Final metrics will be saved in `results_2d_unet/results.csv`

### Memory Considerations
- The script is designed to handle memory constraints
- For systems with limited RAM, consider:
  - Further reducing batch size
  - Processing fewer samples
  - Using a subset of the data for initial experiments 
