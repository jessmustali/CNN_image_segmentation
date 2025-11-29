"""
Custom loss functions and metrics for vocal tract segmentation.
"""

import tensorflow as tf


def cross_entropy(num_classes):
    """
    Custom cross-entropy loss function.
    
    Args:
        num_classes: Number of segmentation classes
    
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        """
        Calculate cross-entropy loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Cross-entropy loss value
        """
        # Determine axis based on input dimensionality
        if len(y_pred.shape) == 4:  # 2D image
            axis = (1, 2)
        elif len(y_pred.shape) == 5:  # 3D volume
            axis = (1, 2, 3)
        
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        loss_image = 0.0
        
        # Calculate loss for each class
        for c in range(num_classes):
            class_loss = -1 * tf.math.reduce_mean(
                tf.math.multiply(y_true[..., c], tf.math.log(y_pred[..., c] + 1e-9)), 
                axis=axis
            )
            loss_image = tf.math.add(class_loss, loss_image)
        
        loss_ce = tf.math.reduce_mean(loss_image)
        
        return loss_ce
    
    return loss


class Mean_DICE(tf.keras.metrics.Metric):
    """
    Mean Dice coefficient metric for segmentation evaluation.
    
    The Dice coefficient measures the overlap between predicted and ground truth segmentations.
    Formula: DICE = (2 * |S_pred ∩ S_true|) / (|S_pred| + |S_true|)
    """
    
    def __init__(self, num_classes, name='Mean_DICE', smooth_factor=1e-9, **kwargs):
        """
        Initialize Mean_DICE metric.
        
        Args:
            num_classes: Number of segmentation classes
            name: Name of the metric
            smooth_factor: Small value to avoid division by zero
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.axis = None
        self.smooth_factor = smooth_factor
        self.mean_dice = self.add_weight(name='mean_dice', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sample_weight: Optional sample weights
        """
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        dice_classes = 0.0
        
        # Determine axis based on input dimensionality
        if len(y_pred.shape) == 4:  # 2D image
            self.axis = (1, 2)
        elif len(y_pred.shape) == 5:  # 3D volume
            self.axis = (1, 2, 3)
        
        # Calculate Dice coefficient for each class
        for c in range(self.num_classes):
            # Sum of true labels for class c
            abs_label = tf.math.reduce_sum(y_true[..., c], axis=self.axis)
            # Sum of predicted labels for class c
            abs_pred = tf.math.reduce_sum(y_pred[..., c], axis=self.axis)
            
            # Denominator: |S_pred| + |S_true|
            MD_batch_denom = tf.math.add(
                tf.math.add(abs_label, abs_pred), 
                self.smooth_factor
            )
            
            # Numerator: 2 * |S_pred ∩ S_true|
            MD_batch_num = tf.math.add(
                2 * tf.math.reduce_sum(
                    tf.math.multiply(y_true[..., c], y_pred[..., c]), 
                    axis=self.axis
                ), 
                self.smooth_factor
            )
            
            # Dice for current class
            dice_image = tf.math.divide(MD_batch_num, MD_batch_denom)
            dice_classes = tf.math.add(dice_image, dice_classes)
        
        # Average across all classes
        mean_dice_value = tf.math.divide(dice_classes, self.num_classes)
        mean_dice_value = tf.math.reduce_mean(mean_dice_value)
        
        self.mean_dice.assign(mean_dice_value)
    
    def result(self):
        """Return the current metric value."""
        return self.mean_dice
    
    def reset_state(self):
        """Reset the metric state."""
        self.mean_dice.assign(0.0)
    
    def get_config(self):
        """Get metric configuration."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'smooth_factor': self.smooth_factor
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create metric from configuration."""
        return cls(**config)


def calculate_mean_dice(y_true, y_pred, num_classes=7, smooth_factor=1e-9):
    """
    Calculate Mean Dice coefficient (standalone function for evaluation).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        smooth_factor: Small value to avoid division by zero
    
    Returns:
        Mean Dice coefficient value
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    dice_classes = 0.0
    
    # Determine axis based on input dimensionality
    if len(y_pred.shape) == 4:
        axis = (1, 2)
    elif len(y_pred.shape) == 5:
        axis = (1, 2, 3)
    else:
        axis = (0, 1)
    
    for c in range(num_classes):
        abs_label = tf.math.reduce_sum(y_true[..., c], axis=axis)
        abs_pred = tf.math.reduce_sum(y_pred[..., c], axis=axis)
        MD_batch_denom = tf.math.add(tf.math.add(abs_label, abs_pred), smooth_factor)
        MD_batch_num = tf.math.add(
            2 * tf.math.reduce_sum(tf.math.multiply(y_true[..., c], y_pred[..., c]), axis=axis), 
            smooth_factor
        )
        dice_image = tf.math.divide(MD_batch_num, MD_batch_denom)
        dice_classes = tf.math.add(dice_image, dice_classes)
    
    mean_dice = tf.math.divide(dice_classes, num_classes)
    mean_dice = tf.math.reduce_mean(mean_dice)
    
    return mean_dice
