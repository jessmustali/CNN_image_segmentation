"""
Training script for vocal tract segmentation model.
"""

import os
import tensorflow as tf
from configs.config import Config
from src.model import get_model
from src.metrics import cross_entropy, Mean_DICE


def compile_model(model, config=None):
    """
    Compile the model with loss function and metrics.
    
    Args:
        model: Keras model to compile
        config: Configuration object
    
    Returns:
        Compiled model
    """
    if config is None:
        config = Config
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            beta_1=config.BETA_1,
            beta_2=config.BETA_2,
            epsilon=config.EPSILON
        ),
        loss=cross_entropy(config.NUM_CLASSES),
        metrics=[Mean_DICE(num_classes=config.NUM_CLASSES)]
    )
    
    return model


def get_callbacks(checkpoint_path, best_model_path, config=None):
    """
    Create training callbacks.
    
    Args:
        checkpoint_path: Path to save checkpoints
        best_model_path: Path to save best model
        config: Configuration object
    
    Returns:
        List of callbacks
    """
    if config is None:
        config = Config
    
    # Model checkpoint callback - saves best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor=config.MONITOR_METRIC,
        mode=config.MONITOR_MODE,
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=2,
        mode=config.MONITOR_MODE,
    )
    
    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=checkpoint_path,
        histogram_freq=1,
        write_graph=True,
    )
    
    return [model_checkpoint, early_stopping, tensorboard_callback]


def train_model(x_train, y_train, x_val, y_val, model=None, config=None, 
                checkpoint_path=None, best_model_path=None):
    """
    Train the segmentation model.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_val: Validation images
        y_val: Validation labels
        model: Model to train (creates new one if None)
        config: Configuration object
        checkpoint_path: Path for checkpoints
        best_model_path: Path for best model
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    if config is None:
        config = Config
    
    # Create model if not provided
    if model is None:
        model = get_model(model_type='standard', config=config)
    
    # Compile model
    model = compile_model(model, config)
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    # Set up callbacks
    if checkpoint_path and best_model_path:
        callbacks = get_callbacks(checkpoint_path, best_model_path, config)
    else:
        callbacks = []
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    return model, history


def grid_search_training(x_train, y_train, x_val, y_val, batch_sizes, dropouts,
                        checkpoint_path, best_model_path, config=None):
    """
    Perform grid search over hyperparameters.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_val: Validation images
        y_val: Validation labels
        batch_sizes: List of batch sizes to try
        dropouts: List of dropout rates to try
        checkpoint_path: Base checkpoint path
        best_model_path: Base best model path
        config: Configuration object
    
    Returns:
        List of training results
    """
    if config is None:
        config = Config
    
    results = []
    iteration = 0
    
    for batch_size in batch_sizes:
        for dropout in dropouts:
            print(f"\n{'='*80}")
            print(f"GRID SEARCH - Iteration {iteration}")
            print(f"Batch size: {batch_size}, Dropout: {dropout}")
            print(f"{'='*80}\n")
            
            # Create directories for this iteration
            current_checkpoint_path = os.path.join(checkpoint_path, f'training_{iteration}')
            current_best_model_path = os.path.join(best_model_path, f'best_model_{iteration}')
            
            os.makedirs(current_checkpoint_path, exist_ok=True)
            os.makedirs(current_best_model_path, exist_ok=True)
            
            # Update config temporarily
            original_batch_size = config.BATCH_SIZE
            original_dropout = config.DROPOUT
            config.BATCH_SIZE = batch_size
            config.DROPOUT = dropout
            
            # Build and train model
            from src.model import build_unet_model
            model = build_unet_model(
                input_shape=config.IMAGE_SIZE,
                num_classes=config.NUM_CLASSES,
                dropout=dropout
            )
            
            model, history = train_model(
                x_train, y_train, x_val, y_val,
                model=model,
                config=config,
                checkpoint_path=current_checkpoint_path,
                best_model_path=current_best_model_path
            )
            
            # Store results
            results.append({
                'iteration': iteration,
                'batch_size': batch_size,
                'dropout': dropout,
                'history': history,
                'model_path': current_best_model_path
            })
            
            # Restore original config
            config.BATCH_SIZE = original_batch_size
            config.DROPOUT = original_dropout
            
            iteration += 1
    
    return results


def load_best_model(model_path, num_classes=7):
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model
        num_classes: Number of classes
    
    Returns:
        Loaded Keras model
    """
    custom_objects = {
        'loss': cross_entropy(num_classes=num_classes),
        'Mean_DICE': Mean_DICE(num_classes=num_classes)
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    return model
