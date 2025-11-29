"""
Inference script for vocal tract segmentation.

Usage:
    python scripts/predict.py --model_path /path/to/model --dataset_path /path/to/dataset
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from src.data_loader import load_dataset, dataset_to_arrays, preprocess_data
from src.train import load_best_model
from src.evaluate import (
    evaluate_model, 
    generate_predictions, 
    evaluate_postprocessing_pipeline,
    detailed_evaluation_report
)
from src.visualization import visualize_predictions, compare_postprocessing_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on vocal tract segmentation model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--min_island_size', type=int, default=12,
                       help='Minimum island size for postprocessing')
    parser.add_argument('--no_denoise', action='store_true',
                       help='Skip denoising preprocessing step')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Update configuration
    Config.DATASET_PATH = args.dataset_path
    
    print("\n" + "="*80)
    print("VOCAL TRACT SEGMENTATION - INFERENCE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Dataset path: {Config.DATASET_PATH}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Min island size: {args.min_island_size}")
    print(f"  Apply denoising: {not args.no_denoise}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Step 4: Load model
    print("\nStep 4: Loading model...")
    model = load_best_model(args.model_path, num_classes=Config.NUM_CLASSES)
    print(f"  Model loaded from: {args.model_path}")
    
    # Step 5: Evaluate on test set
    print("\nStep 5: Evaluating model on test set...")
    test_scores = evaluate_model(model, x_test, y_test)
    
    # Step 6: Generate predictions
    print("\nStep 6: Generating predictions...")
    predictions = generate_predictions(model, x_test)
    
    # Step 7: Postprocess and evaluate
    print("\nStep 7: Postprocessing predictions...")
    postprocessing_results = evaluate_postprocessing_pipeline(
        y_test,
        predictions,
        min_island_size=args.min_island_size,
        apply_filtering=True,
        filter_size=3,
        num_classes=Config.NUM_CLASSES
    )
    
    # Step 8: Detailed evaluation
    print("\nStep 8: Generating detailed evaluation report...")
    best_predictions = postprocessing_results['predictions'][postprocessing_results['best_method']]
    
    class_names = [
        'Background',
        'Soft Palate',
        'Jaw',
        'Tongue',
        'Vocal Tract',
        'Tooth Space',
        'Head'
    ]
    
    detailed_report = detailed_evaluation_report(
        y_test,
        best_predictions,
        num_classes=Config.NUM_CLASSES,
        class_names=class_names
    )
    
    # Step 9: Visualize results
    if args.visualize:
        print("\nStep 9: Visualizing predictions...")
        
        # Visualize raw predictions
        print("  Visualizing raw predictions...")
        visualize_predictions(
            x_test, y_test, predictions,
            num_samples=args.num_samples,
            num_classes=Config.NUM_CLASSES
        )
        
        # Compare postprocessing methods
        print("  Comparing postprocessing methods...")
        for i in range(min(args.num_samples, len(x_test))):
            compare_postprocessing_results(
                y_test,
                postprocessing_results['predictions'],
                sample_idx=i
            )
    
    # Step 10: Save results summary
    print("\nStep 10: Saving results...")
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VOCAL TRACT SEGMENTATION - EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {Config.DATASET_PATH}\n\n")
        
        f.write("Test Set Evaluation:\n")
        f.write(f"  Loss: {test_scores[0]:.4f}\n")
        f.write(f"  Mean DICE: {test_scores[1]:.4f}\n\n")
        
        f.write("Postprocessing Results:\n")
        for method, score in postprocessing_results['scores'].items():
            f.write(f"  {method}: {score:.4f}\n")
        
        f.write(f"\nBest Method: {postprocessing_results['best_method']}\n")
        f.write(f"Best Score: {postprocessing_results['best_score']:.4f}\n\n")
        
        f.write("Per-Class DICE Coefficients:\n")
        for name, dice in zip(class_names, detailed_report['per_class_dice']):
            f.write(f"  {name}: {dice:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nBest postprocessing method: {postprocessing_results['best_method']}")
    print(f"Best Mean DICE score: {postprocessing_results['best_score']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
