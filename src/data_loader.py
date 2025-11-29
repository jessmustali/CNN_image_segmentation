"""
Data loading and preprocessing utilities.
"""

import numpy as np
from scipy import ndimage
import med_dataloader as mdl
from configs.config import Config


def load_dataset(data_dir=None, config=None):
    """
    Load dataset using med_dataloader.
    
    Args:
        data_dir: Path to dataset directory
        config: Configuration object (uses Config if None)
    
    Returns:
        Tuple of (train_ds, validation_ds, test_ds) as tf.data.Dataset objects
    """
    if config is None:
        config = Config
    
    if data_dir is None:
        data_dir = config.DATASET_PATH
    
    train_ds, validation_ds, test_ds = mdl.get_dataset(
        data_dir=data_dir,
        percentages=[config.PERC_TRAIN_DATA, config.PERC_VALIDATION_DATA, config.PERC_TEST_DATA],
        batch_size=config.BATCH_SIZE,
        train_augmentation=config.DATA_AUGMENTATION,
        random_crop_size=config.CROP_SIZE,
        random_rotate=config.ROTATION,
        random_flip=config.FLIP,
    )
    
    return train_ds, validation_ds, test_ds


def dataset_to_arrays(train_ds, validation_ds, test_ds):
    """
    Convert tf.data.Dataset objects to numpy arrays.
    
    Args:
        train_ds: Training dataset
        validation_ds: Validation dataset
        test_ds: Test dataset
    
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    
    # Extract training data
    for img, lbl in train_ds.unbatch():
        x_train.append(img)
        y_train.append(lbl)
    
    # Extract validation data
    for img, lbl in validation_ds.unbatch():
        x_val.append(img)
        y_val.append(lbl)
    
    # Extract test data
    for img, lbl in test_ds.unbatch():
        x_test.append(img)
        y_test.append(lbl)
    
    # Convert to numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    print(f'Total images: {len(x_train) + len(x_val) + len(x_test)}')
    print(f'x_train shape: {x_train.shape}  y_train shape: {y_train.shape}')
    print(f'x_val shape: {x_val.shape}  y_val shape: {y_val.shape}')
    print(f'x_test shape: {x_test.shape}  y_test shape: {y_test.shape}')
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def apply_median_filter(data, filter_size=3):
    """
    Apply median filter to denoise images.
    
    Args:
        data: Input image array of shape (n_samples, height, width, channels)
        filter_size: Size of the median filter kernel
    
    Returns:
        Denoised image array
    """
    data_denoised = np.zeros(data.shape)
    
    for n, volume in enumerate(data):
        data_denoised[n] = ndimage.median_filter(volume, filter_size)
    
    return data_denoised


def preprocess_data(x_train, x_val, x_test, apply_denoising=True, filter_size=3):
    """
    Preprocess data by applying median filtering for denoising.
    
    Args:
        x_train: Training images
        x_val: Validation images
        x_test: Test images
        apply_denoising: Whether to apply median filtering
        filter_size: Size of median filter kernel
    
    Returns:
        Tuple of (x_train_processed, x_val_processed, x_test_processed)
    """
    if apply_denoising:
        print("Applying median filter for denoising...")
        x_train_processed = apply_median_filter(x_train, filter_size)
        x_val_processed = apply_median_filter(x_val, filter_size)
        x_test_processed = apply_median_filter(x_test, filter_size)
        
        print(f'x_train_denoised shape: {x_train_processed.shape}')
        print(f'x_val_denoised shape: {x_val_processed.shape}')
        print(f'x_test_denoised shape: {x_test_processed.shape}')
    else:
        print("Skipping denoising step...")
        x_train_processed = x_train
        x_val_processed = x_val
        x_test_processed = x_test
    
    return x_train_processed, x_val_processed, x_test_processed
