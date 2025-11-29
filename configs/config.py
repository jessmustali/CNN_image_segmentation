"""
Configuration file for vocal tract segmentation project.
Contains all hyperparameters and paths.
"""

import os
from datetime import datetime
from dateutil.tz import gettz


class Config:
    """Main configuration class for the project."""
    
    # Dataset parameters
    NUM_CLASSES = 7
    IMAGE_SIZE = (256, 256, 1)
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 70
    LEARNING_RATE = 1e-3
    
    # Data augmentation parameters
    DATA_AUGMENTATION = False
    ROTATION = True
    FLIP = True
    CROP_SIZE = True
    
    # Dataset split percentages
    PERC_TRAIN_DATA = 0.8
    PERC_VALIDATION_DATA = 0.1
    PERC_TEST_DATA = 0.1
    
    # Model architecture parameters
    FILTERS = 64
    DROPOUT = 0.3
    USE_BATCH_NORM = True
    
    # Optimizer parameters
    BETA_1 = 0.9
    BETA_2 = 0.999
    EPSILON = 1e-08
    
    # Callback parameters
    EARLY_STOPPING_PATIENCE = 8
    MONITOR_METRIC = 'val_Mean_DICE'
    MONITOR_MODE = 'max'
    
    # Preprocessing parameters
    MEDIAN_FILTER_SIZE = 3
    
    # Postprocessing parameters
    MIN_ISLAND_SIZE = 12
    
    # Paths (should be overridden with actual paths)
    ROOT_PATH = os.path.join(os.sep, 'content', 'gdrive')
    WD_PATH = None
    DATASET_PATH = None
    MODELS_PATH = None
    
    @classmethod
    def set_paths(cls, root_path=None, working_dir='vocal_tract_segmentation'):
        """Set up directory paths."""
        if root_path:
            cls.ROOT_PATH = root_path
        
        cls.WD_PATH = os.path.join(cls.ROOT_PATH, 'MyDrive', working_dir)
        cls.DATASET_PATH = os.path.join(cls.WD_PATH, 'dataset_vocal_tract_SP')
        cls.MODELS_PATH = os.path.join(cls.WD_PATH, 'my_models')
    
    @classmethod
    def get_training_path(cls):
        """Generate a new training path with timestamp."""
        timestamp = datetime.now(gettz("Europe/Rome")).strftime("%Y-%m-%d-T%H:%M")
        training_path = os.path.join(cls.MODELS_PATH, f'Training_{timestamp}')
        checkpoint_path = os.path.join(training_path, 'Checkpoints')
        best_model_path = os.path.join(training_path, 'Best Model')
        
        return training_path, checkpoint_path, best_model_path
    
    @classmethod
    def create_directories(cls, training_path, checkpoint_path, best_model_path):
        """Create necessary directories for training."""
        for path in [training_path, checkpoint_path, best_model_path]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)


# Grid search parameters (optional)
class GridSearchConfig:
    """Configuration for hyperparameter grid search."""
    
    BATCH_SIZES = [4, 32]
    DROPOUTS = [0, 0.25, 0.5]
    FILTERS_OPTIONS = [64]
