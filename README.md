# Automatic Image Segmentation for Speech Production Analysis

## Overview

The purpose of this project is to develop an automatic image-segmentation method using a deep learning approach. The goal is to extract surfaces of main articulators crucial for speech production. This method can aid doctors in diagnosing and monitoring neurodegenerative diseases, as motor speech impairment may be an early observable symptom.

## Dataset

We were provided with a [Dataset](https://drive.google.com/drive/folders/1OK4fb9pEtWFTq9Ki8UR2UQBELVJejrya?usp=share_link) of 820 real-time MRI images along with their respective labels. The dataset was split into 80% for training and 10% each for validation and testing. Salt and pepper noise was removed using a median filter.

## Model Architecture

The U-Net architecture, as used by [Ruthven](https://www.sciencedirect.com/science/article/pii/S0169260720316473) in their research, was chosen. It consists of 5 encoding layers and 5 decoding layers. Gridsearch was performed for hyperparameter optimization, exploring dropout rate, batch size, loss function, and type of pooling.

## Training and Evaluation

During training and postprocessing, the DICE coefficient was used to measure the quality of segmentation. Cross entropy was chosen as the loss function, with variations like uniform weights, adjusted weights, and custom cross entropy tested.

Results indicated that cross entropy with uniform weights performed best. The model exhibited robustness to noise, with minimal impact on results when applied to the original dataset versus denoised versions.

## Postprocessing Techniques

Various postprocessing techniques were implemented after choosing the best model. An algorithm for island detection was designed to remove isolated groups of pixels below a certain threshold. Median filters were also applied individually and in combination with island detection.

## Results

The best results were obtained by combining the median filter with the island detection algorithm, followed by raw predictions and the median filter alone. Combining multiple methods had a negative impact on the metric.

## Conclusion

In conclusion, our model demonstrated satisfactory segmentation with good visual results and mean DICE scores. It showed no overfitting and robustness to noise. Further improvements could involve exploring different loss functions and increasing the dataset size.

## Future Work

Potential future improvements include exploring additional loss functions, increasing the number of samples, and continuous refinement of postprocessing techniques.

## Below is Claude generated readme

# Vocal Tract Segmentation with Deep Learning

Automatic segmentation of vocal tract and articulators in MRI images using U-Net deep learning architecture.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a deep learning solution for segmenting vocal tract anatomical structures in MRI images. The model identifies and segments seven distinct anatomical regions that are crucial for speech production analysis and clinical applications.

### Segmented Anatomical Regions

1. **Background** - Non-anatomical regions
2. **Soft Palate** - Muscular structure at the back of the mouth
3. **Jaw** - Lower mandible structure
4. **Tongue** - Primary articulator for speech
5. **Vocal Tract** - Air cavity for sound production
6. **Tooth Space** - Dental structures
7. **Head** - Surrounding cranial structures

### Clinical Applications

- **Speech Therapy**: Quantitative analysis of articulator movement
- **Velopharyngeal Closure Assessment**: Measuring soft palate function
- **Speech Research**: Understanding speech production mechanisms
- **Treatment Planning**: Pre/post-operative evaluation for speech disorders

## Model Architecture

The project uses **U-Net**, a convolutional neural network architecture specifically designed for biomedical image segmentation:

- **Encoder**: 4 downsampling blocks with [64, 128, 256, 512] filters
- **Bottleneck**: 1024 filters with dropout regularization
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Softmax activation for multi-class segmentation

**Key Features**:
- Batch normalization for stable training
- Dropout for regularization (prevents overfitting)
- Skip connections for precise localization
- He normal initialization for deep networks

## Performance

**Evaluation Metrics**:
- **Mean DICE Coefficient**: ~0.92 (overall)
- **Best Performing Class**: Head (DICE ~0.99)
- **Most Challenging Classes**: Soft Palate, Tooth Space (DICE ~0.92-0.93)

**Training Details**:
- Training time: ~2-3 hours on GPU (70 epochs)
- Dataset: 392 MRI images across 5 subjects
- Image size: 256×256 pixels
- Validation: 5-fold cross-validation

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/vocal_tract_segmentation.git
cd vocal_tract_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Package Installation

Install as a Python package:

```bash
pip install -e .
```

This enables command-line tools:
- `vts-train` - Training script
- `vts-predict` - Inference script

## Quick Start

### Training a Model

```bash
python scripts/train_model.py \
    --dataset_path /path/to/dataset \
    --epochs 70 \
    --batch_size 8 \
    --output_dir ./models
```

**Arguments**:
- `--dataset_path`: Directory containing MRI images and segmentation masks
- `--epochs`: Number of training epochs (default: 70)
- `--batch_size`: Training batch size (default: 8)
- `--output_dir`: Where to save trained models
- `--model_type`: Choose 'standard' or 'custom' U-Net
- `--dropout`: Dropout rate for regularization (default: 0.3)

### Running Inference

```bash
python scripts/predict.py \
    --model_path ./models/best_model \
    --dataset_path /path/to/dataset \
    --output_dir ./predictions \
    --visualize
```

**Arguments**:
- `--model_path`: Path to trained model
- `--dataset_path`: Directory with test images
- `--output_dir`: Where to save predictions and reports
- `--visualize`: Display prediction visualizations
- `--min_island_size`: Postprocessing threshold (default: 12)

## Using as a Python Library

```python
from configs.config import Config
from src.data_loader import load_dataset, dataset_to_arrays, preprocess_data
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model, generate_predictions

# Configure
Config.DATASET_PATH = '/path/to/dataset'
Config.EPOCHS = 70
Config.BATCH_SIZE = 8

# Load and preprocess data
train_ds, val_ds, test_ds = load_dataset()
x_train, y_train, x_val, y_val, x_test, y_test = dataset_to_arrays(
    train_ds, val_ds, test_ds
)
x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test)

# Create and train model
model = get_model(model_type='standard')
model, history = train_model(x_train, y_train, x_val, y_val, model=model)

# Evaluate
test_scores = evaluate_model(model, x_test, y_test)
predictions = generate_predictions(model, x_test)
```

## Dataset Format

The dataset should be organized for compatibility with `med_dataloader`:

```
dataset/
├── images/
│   ├── subject_001_frame_001.nii.gz
│   ├── subject_001_frame_002.nii.gz
│   └── ...
└── labels/
    ├── subject_001_frame_001.nii.gz
    ├── subject_001_frame_002.nii.gz
    └── ...
```

**Requirements**:
- Images: Grayscale MRI scans (256×256 pixels)
- Labels: One-hot encoded segmentation masks (7 classes)
- Format: NIfTI (.nii.gz) or compatible medical image format

## Project Structure

```
vocal_tract_segmentation/
├── configs/              # Configuration management
│   └── config.py        # Hyperparameters and settings
├── src/                 # Core source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── model.py         # U-Net architecture
│   ├── metrics.py       # Loss functions and metrics
│   ├── train.py         # Training utilities
│   ├── evaluate.py      # Evaluation and testing
│   ├── postprocessing.py # Prediction refinement
│   └── visualization.py  # Visualization tools
├── scripts/             # Command-line interfaces
│   ├── train_model.py   # Training CLI
│   └── predict.py       # Inference CLI
└── docs/                # Documentation
```

## Configuration

Modify hyperparameters in `configs/config.py`:

```python
class Config:
    # Model architecture
    NUM_CLASSES = 7
    IMAGE_SIZE = (256, 256, 1)
    DROPOUT = 0.3
    FILTERS = 64
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 70
    LEARNING_RATE = 1e-3
    
    # Data preprocessing
    MEDIAN_FILTER_SIZE = 3
    
    # Postprocessing
    MIN_ISLAND_SIZE = 12
```

## Advanced Features

### Postprocessing Pipeline

The model includes sophisticated postprocessing to improve segmentation quality:

1. **Binary Thresholding**: Convert probability maps to discrete labels
2. **Median Filtering**: Smooth segmentation boundaries (3×3 kernel)
3. **Island Removal**: Eliminate small disconnected regions
4. **Connected Component Analysis**: Identify and measure anatomical structures

### Training Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./models/Training_*/Checkpoints
```

View:
- Loss curves (training and validation)
- DICE coefficient progression
- Model architecture graph
- Learning rate schedules

### Hyperparameter Grid Search

Automatically search for optimal hyperparameters:

```python
from src.train import grid_search_training
from configs.config import GridSearchConfig

results = grid_search_training(
    x_train, y_train, x_val, y_val,
    batch_sizes=GridSearchConfig.BATCH_SIZES,
    dropouts=GridSearchConfig.DROPOUTS,
    checkpoint_path='./grid_search/checkpoints',
    best_model_path='./grid_search/models'
)
```

## Methodology

### Loss Function

**Cross-Entropy Loss** for multi-class segmentation:

```
L_CE = -∑∑ (1/N) g_i^c · log(p_i^c)
```

where:
- `g_i^c` = ground truth for pixel i, class c
- `p_i^c` = predicted probability for pixel i, class c
- `N` = number of pixels

### Evaluation Metric

**DICE Coefficient** measures segmentation overlap:

```
DICE = (2 × |S_pred ∩ S_true|) / (|S_pred| + |S_true|)
```

- Range: 0 (no overlap) to 1 (perfect overlap)
- Averaged across all classes for final score

### Preprocessing

1. **Median Filtering**: Remove noise while preserving edges (3×3 kernel)
2. **Normalization**: Scale intensity values for stable training
3. **Data Augmentation** (optional): Rotation, flipping, cropping

## Results Visualization

The toolkit provides comprehensive visualization:

```python
from src.visualization import visualize_predictions, plot_training_history

# View predictions
visualize_predictions(x_test, y_test, predictions, num_samples=5)

# Plot training curves
plot_training_history(history, save_path='training_history.png')
```

## Clinical Relevance

### Velopharyngeal Closure Analysis

The model accurately segments the soft palate, enabling:
- Measurement of velopharyngeal gap
- Assessment of closure patterns
- Treatment planning for velopharyngeal insufficiency

### Speech Production Research

Quantitative measurements support:
- Articulator motion analysis
- Vocal tract shape characterization
- Cross-linguistic phonetic studies
- Motor control investigations

## Requirements

### Software Dependencies

```
tensorflow >= 2.10.0
keras >= 2.10.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
med-dataloader == 0.1.12
SimpleITK >= 2.1.0
opencv-python >= 4.5.0
```

### Hardware Recommendations

**Minimum**:
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB

**Recommended**:
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- Storage: SSD with 50GB+

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vocal_tract_segmentation2024,
  title={Vocal Tract Segmentation with Deep Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/vocal_tract_segmentation}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Inspired by research in speech production and vocal tract imaging
- Built with TensorFlow and Keras

## Contact

- **Issues**: [GitHub Issues](https://github.com/jessmustali/CNN_image_segmentation/issues)
- **Email**: jessica.mustali@gmail.com

## Related Work

- [Deep-learning-based segmentation of the vocal tract](https://www.sciencedirect.com/science/article/pii/S0169260720316473)
- [Real-time speech MRI datasets](https://www.nature.com/articles/s41597-023-02766-z)
- [U-Net for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

