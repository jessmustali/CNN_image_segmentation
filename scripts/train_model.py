"""
Main training script for vocal tract segmentation.

Usage:
    python scripts/train_model.py --dataset_path /path/to/dataset --epochs 70
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from src.data_loader import load_dataset, dataset_to_arrays, preprocess_data
from src.model import get_model
from src.train import train_model, compile_model, get_callbacks
from src.visualization import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train vocal tract segmentation model')
    
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for saved models')
    parser.add_argument('--epochs', type=int, default=70,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'custom'],
                       help='Type of U-Net model to use')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--no_denoise', action='store_true',
                       help='Skip denoising preprocessing step')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Update configuration
    Config.DATASET_PATH = args.dataset_path
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.DROPOUT = args.dropout
    Config.MODELS_PATH = args.output_dir
    
    print("\n" + "="*80)
    print("VOCAL TRACT SEGMENTATION - TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset path: {Config.DATASET_PATH}")
    print(f"  Output directory: {Config.MODELS_PATH}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Dropout: {Config.DROPOUT}")
    print(f"  Model type: {args.model_type}")
    print(f"  Apply denoising: {not args.no_denoise}")
    print("="*80 + "\n")
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    train_ds, validation_ds, test_ds = load_dataset(config=Config)
    
    # Step 2: Convert to numpy arrays
    print("\nStep 2: Converting to numpy arrays...")
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_to_arrays(
        train_ds, validation_ds, test_ds
    )
    
    # Step 3: Preprocess data
    print("\nStep 3: Preprocessing data...")
    x_train, x_val, x_test = preprocess_data(
        x_train, x_val, x_test,
        apply_denoising=not args.no_denoise
    )
    
    # Step 4: Create model
    print("\nStep 4: Creating model...")
    model = get_model(model_type=args.model_type, config=Config)
    
    # Step 5: Setup training paths
    print("\nStep 5: Setting up training directories...")
    training_path, checkpoint_path, best_model_path = Config.get_training_path()
    Config.create_directories(training_path, checkpoint_path, best_model_path)
    
    print(f"  Training path: {training_path}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Best model: {best_model_path}")
    
    # Step 6: Train model
    print("\nStep 6: Training model...")
    model, history = train_model(
        x_train, y_train, x_val, y_val,
        model=model,
        config=Config,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path
    )
    
    # Step 7: Plot and save training history
    print("\nStep 7: Saving training history...")
    plot_path = os.path.join(training_path, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    print(f"  Training history saved to: {plot_path}")
    
    # Step 8: Save final model
    print("\nStep 8: Saving final model...")
    final_model_path = os.path.join(training_path, 'final_model')
    model.save(final_model_path)
    print(f"  Final model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {best_model_path}")
    print(f"Training path: {training_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
