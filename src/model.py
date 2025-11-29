"""
U-Net model architecture for vocal tract segmentation.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.config import Config


def double_conv_block(x, n_filters):
    """
    Double convolution block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    
    Args:
        x: Input tensor
        n_filters: Number of filters
    
    Returns:
        Output tensor after double convolution
    """
    x = layers.Conv2D(n_filters, 3, padding="same", activation="linear", 
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(n_filters, 3, padding="same", activation="linear", 
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    return x


def downsample_block(x, n_filters):
    """
    Downsampling block: double_conv -> MaxPooling
    
    Args:
        x: Input tensor
        n_filters: Number of filters
    
    Returns:
        Tuple of (features, pooled_output)
    """
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    return f, p


def upsample_block(x, conv_features, n_filters):
    """
    Upsampling block: Transpose Conv -> Concatenate -> double_conv
    
    Args:
        x: Input tensor
        conv_features: Skip connection features from encoder
        n_filters: Number of filters
    
    Returns:
        Output tensor after upsampling
    """
    # Upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # Concatenate with skip connection
    x = layers.concatenate([x, conv_features])
    # Apply double convolution
    x = double_conv_block(x, n_filters)
    
    return x


def build_unet_model(input_shape=(256, 256, 1), num_classes=7, dropout=0.3):
    """
    Build U-Net model for image segmentation.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
        dropout: Dropout rate
    
    Returns:
        Keras Model object
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    p4 = layers.Dropout(dropout)(p4)
    
    # Bottleneck
    bottleneck = double_conv_block(p4, 1024)
    bottleneck = layers.Dropout(dropout)(bottleneck)
    
    # Decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(u9)
    
    # Create model
    unet_model = Model(inputs, outputs, name="U-Net")
    
    return unet_model


def build_custom_unet_model(input_shape=(256, 256, 1), num_classes=7, 
                           filters=64, dropout=0.4, use_batch_norm=True):
    """
    Build custom U-Net model using keras-unet library.
    Requires: pip install keras-unet
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of segmentation classes
        filters: Base number of filters
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    
    Returns:
        Keras Model object
    """
    try:
        from keras_unet.models import custom_unet
        
        model = custom_unet(
            input_shape=input_shape,
            use_batch_norm=use_batch_norm,
            num_classes=num_classes,
            filters=filters,
            dropout=dropout,
            output_activation='softmax'
        )
        
        return model
    
    except ImportError:
        print("keras-unet not installed. Install with: pip install keras-unet")
        print("Falling back to standard U-Net model...")
        return build_unet_model(input_shape, num_classes, dropout)


def get_model(model_type='standard', config=None):
    """
    Get model based on type.
    
    Args:
        model_type: 'standard' or 'custom'
        config: Configuration object
    
    Returns:
        Compiled or uncompiled Keras model
    """
    if config is None:
        config = Config
    
    if model_type == 'custom':
        model = build_custom_unet_model(
            input_shape=config.IMAGE_SIZE,
            num_classes=config.NUM_CLASSES,
            filters=config.FILTERS,
            dropout=config.DROPOUT,
            use_batch_norm=config.USE_BATCH_NORM
        )
    else:
        model = build_unet_model(
            input_shape=config.IMAGE_SIZE,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
    
    return model
