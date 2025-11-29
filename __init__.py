"""
Vocal Tract Segmentation Package

A deep learning toolkit for segmenting vocal tract MRI images using U-Net.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from src.model import build_unet_model, get_model
from src.metrics import Mean_DICE, cross_entropy, calculate_mean_dice
from src.train import train_model, compile_model, load_best_model
from src.evaluate import evaluate_model, generate_predictions
from src.data_loader import load_dataset, dataset_to_arrays, preprocess_data
from src.postprocessing import postprocess_predictions
from src.visualization import visualize_predictions, plot_training_history

__all__ = [
    'build_unet_model',
    'get_model',
    'Mean_DICE',
    'cross_entropy',
    'calculate_mean_dice',
    'train_model',
    'compile_model',
    'load_best_model',
    'evaluate_model',
    'generate_predictions',
    'load_dataset',
    'dataset_to_arrays',
    'preprocess_data',
    'postprocess_predictions',
    'visualize_predictions',
    'plot_training_history',
]
