"""
Evaluation utilities for model testing and analysis.
"""

import numpy as np
import tensorflow as tf
from src.metrics import calculate_mean_dice
from src.postprocessing import postprocess_predictions


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
    
    Returns:
        Evaluation scores (loss and metrics)
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60 + "\n")
    
    test_scores = model.evaluate(x=x_test, y=y_test, verbose=2)
    
    print("\n" + "="*60)
    print(f"Test Loss: {test_scores[0]:.4f}")
    print(f"Test Mean DICE: {test_scores[1]:.4f}")
    print("="*60 + "\n")
    
    return test_scores


def generate_predictions(model, x_test):
    """
    Generate predictions for test set.
    
    Args:
        model: Trained model
        x_test: Test images
    
    Returns:
        Predictions array
    """
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60 + "\n")
    
    predictions = model.predict(x_test)
    
    print(f"Predictions shape: {predictions.shape}")
    
    return predictions


def evaluate_predictions(y_test, predictions_dict, num_classes=7):
    """
    Evaluate different postprocessing strategies.
    
    Args:
        y_test: Ground truth labels
        predictions_dict: Dictionary of predictions from different postprocessing methods
        num_classes: Number of classes
    
    Returns:
        Dictionary of scores for each method
    """
    scores = {}
    
    print("\n" + "="*60)
    print("EVALUATING POSTPROCESSING METHODS")
    print("="*60 + "\n")
    
    for name, predictions in predictions_dict.items():
        # Calculate mean DICE for all images
        score_sum = 0.0
        for k in range(y_test.shape[0]):
            mean_dice = calculate_mean_dice(
                y_test[k, :, :, :],
                predictions[k, :, :, :],
                num_classes=num_classes
            )
            score_sum += mean_dice
        
        avg_score = score_sum / y_test.shape[0]
        scores[name] = float(avg_score)
        
        print(f"{name:30s}: Mean DICE = {avg_score:.4f}")
    
    print("\n" + "="*60 + "\n")
    
    return scores


def evaluate_postprocessing_pipeline(y_test, raw_predictions, min_island_size=12,
                                     apply_filtering=True, filter_size=3, num_classes=7):
    """
    Complete evaluation pipeline for postprocessing methods.
    
    Args:
        y_test: Ground truth labels
        raw_predictions: Raw model predictions
        min_island_size: Minimum island size threshold
        apply_filtering: Whether to apply median filtering
        filter_size: Size of median filter
        num_classes: Number of classes
    
    Returns:
        Dictionary with postprocessed predictions and their scores
    """
    # Apply postprocessing
    print("Applying postprocessing methods...")
    postprocessed = postprocess_predictions(
        raw_predictions,
        min_island_size=min_island_size,
        apply_filtering=apply_filtering,
        filter_size=filter_size,
        num_classes=num_classes
    )
    
    # Evaluate each method
    scores = evaluate_predictions(y_test, postprocessed, num_classes)
    
    # Find best method
    best_method = max(scores, key=scores.get)
    best_score = scores[best_method]
    
    print("\n" + "="*60)
    print(f"BEST POSTPROCESSING METHOD: {best_method}")
    print(f"BEST MEAN DICE SCORE: {best_score:.4f}")
    print("="*60 + "\n")
    
    return {
        'predictions': postprocessed,
        'scores': scores,
        'best_method': best_method,
        'best_score': best_score
    }


def calculate_per_class_dice(y_true, y_pred, num_classes=7):
    """
    Calculate Dice coefficient for each class separately.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Array of Dice coefficients per class
    """
    dice_per_class = np.zeros(num_classes)
    
    for c in range(num_classes):
        intersection = np.sum(y_true[:, :, :, c] * y_pred[:, :, :, c])
        sum_true = np.sum(y_true[:, :, :, c])
        sum_pred = np.sum(y_pred[:, :, :, c])
        
        dice = (2.0 * intersection + 1e-9) / (sum_true + sum_pred + 1e-9)
        dice_per_class[c] = dice
    
    return dice_per_class


def detailed_evaluation_report(y_test, predictions, num_classes=7, class_names=None):
    """
    Generate detailed evaluation report with per-class metrics.
    
    Args:
        y_test: Ground truth labels
        predictions: Model predictions
        num_classes: Number of classes
        class_names: Optional list of class names
    
    Returns:
        Dictionary with detailed metrics
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Overall metrics
    overall_dice = calculate_mean_dice(y_test, predictions, num_classes)
    
    # Per-class metrics
    per_class_dice = calculate_per_class_dice(y_test, predictions, num_classes)
    
    print("\n" + "="*60)
    print("DETAILED EVALUATION REPORT")
    print("="*60 + "\n")
    print(f"Overall Mean DICE: {float(overall_dice):.4f}\n")
    print("Per-Class DICE Coefficients:")
    print("-" * 60)
    
    for i, (name, dice) in enumerate(zip(class_names, per_class_dice)):
        print(f"{name:20s}: {dice:.4f}")
    
    print("="*60 + "\n")
    
    return {
        'overall_dice': float(overall_dice),
        'per_class_dice': per_class_dice,
        'class_names': class_names
    }
