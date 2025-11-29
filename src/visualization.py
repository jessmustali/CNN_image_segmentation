"""
Visualization utilities for displaying images and predictions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def visualize_dataset(x_data, y_data, num_classes=7, n_samples=5):
    """
    Visualize images and their corresponding labels.
    
    Args:
        x_data: Input images
        y_data: Label masks
        num_classes: Number of segmentation classes
        n_samples: Number of samples to visualize
    """
    n = 0
    for volume, label in zip(x_data[:n_samples], y_data[:n_samples]):
        print(f"\033[92m\033[1m Image n°: {n} ↓ \033[0m")
        
        fig = plt.figure(figsize=(23, 7))
        
        # Plot original image
        plt.subplot(1, num_classes + 1, 1)
        plt.imshow(volume[:, :, 0], cmap="gray")
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot each class mask
        for i in range(num_classes):
            plt.subplot(1, num_classes + 1, i + 2)
            plt.imshow(label[:, :, i])
            plt.title(f"Class {i}")
            plt.axis('off')
        
        n += 1
        plt.tight_layout()
        plt.show()


def visualize_predictions(x_test, y_test, predictions, num_samples=5, num_classes=7):
    """
    Visualize test images with ground truth and predictions.
    
    Args:
        x_test: Test images
        y_test: Ground truth labels
        predictions: Model predictions
        num_samples: Number of samples to display
        num_classes: Number of classes
    """
    for idx in range(min(num_samples, len(x_test))):
        fig = plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(x_test[idx, :, :, 0], cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        gt_colored = create_colored_mask(y_test[idx], num_classes)
        plt.imshow(gt_colored)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 3, 3)
        pred_colored = create_colored_mask(predictions[idx], num_classes)
        plt.imshow(pred_colored)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def create_colored_mask(mask, num_classes):
    """
    Convert one-hot encoded mask to colored RGB image.
    
    Args:
        mask: One-hot encoded mask of shape (height, width, num_classes)
        num_classes: Number of classes
    
    Returns:
        RGB colored mask
    """
    # Define colors for each class
    colors = [
        [0, 0, 0],       # Background - black
        [255, 0, 0],     # Class 1 - red
        [0, 255, 0],     # Class 2 - green
        [0, 0, 255],     # Class 3 - blue
        [255, 255, 0],   # Class 4 - yellow
        [255, 0, 255],   # Class 5 - magenta
        [0, 255, 255],   # Class 6 - cyan
    ]
    
    # Ensure we have enough colors
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    # Create colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            class_idx = np.argmax(mask[i, j, :])
            colored_mask[i, j, :] = colors[class_idx]
    
    return colored_mask


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Mean DICE
    metric_key = 'Mean_DICE'
    if metric_key in history.history:
        axes[1].plot(history.history[metric_key], label='Training Mean DICE')
        if f'val_{metric_key}' in history.history:
            axes[1].plot(history.history[f'val_{metric_key}'], label='Validation Mean DICE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean DICE')
        axes[1].set_title('Training and Validation Mean DICE')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_postprocessing_results(y_test, results_dict, sample_idx=0):
    """
    Compare different postprocessing results side by side.
    
    Args:
        y_test: Ground truth labels
        results_dict: Dictionary with different postprocessing results
        sample_idx: Index of sample to visualize
    """
    n_results = len(results_dict) + 1  # +1 for ground truth
    fig = plt.figure(figsize=(5 * n_results, 5))
    
    # Plot ground truth
    plt.subplot(1, n_results, 1)
    gt_colored = create_colored_mask(y_test[sample_idx], y_test.shape[-1])
    plt.imshow(gt_colored)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot each postprocessing result
    for idx, (name, result) in enumerate(results_dict.items(), start=2):
        plt.subplot(1, n_results, idx)
        result_colored = create_colored_mask(result[sample_idx], result.shape[-1])
        plt.imshow(result_colored)
        plt.title(name.replace('_', ' ').title())
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
