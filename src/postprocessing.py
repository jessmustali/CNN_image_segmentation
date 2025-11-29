"""
Postprocessing utilities for prediction refinement.
"""

import numpy as np
from scipy import ndimage


def binarize_predictions(predictions):
    """
    Convert soft predictions to binary (one-hot) format.
    Sets the maximum probability class to 1, all others to 0.
    
    Args:
        predictions: Array of shape (n_samples, height, width, n_classes)
    
    Returns:
        Binary predictions of the same shape
    """
    predictions_binary = np.zeros(predictions.shape)
    
    for image in range(predictions.shape[0]):
        for i in range(predictions.shape[1]):
            for j in range(predictions.shape[2]):
                max_index = np.argmax(predictions[image, i, j, :])
                predictions_binary[image, i, j, max_index] = 1
    
    return predictions_binary


def apply_median_filter_to_predictions(predictions, filter_size=3):
    """
    Apply median filter to predictions for smoothing.
    
    Args:
        predictions: Binary predictions array
        filter_size: Size of median filter kernel
    
    Returns:
        Filtered predictions
    """
    predictions_filtered = np.zeros(predictions.shape)
    
    for image in range(predictions.shape[0]):
        for clas in range(predictions.shape[3]):
            predictions_filtered[image, :, :, clas] = ndimage.median_filter(
                predictions[image, :, :, clas], 
                filter_size
            )
    
    return predictions_filtered


def find_island_size(island_matrix, island_index):
    """
    Calculate the size (number of pixels) of a specific island.
    
    Args:
        island_matrix: Matrix with island labels
        island_index: Index of the island to measure
    
    Returns:
        Size of the island in pixels
    """
    island_size = 0
    for i in range(island_matrix.shape[0]):
        for j in range(island_matrix.shape[1]):
            if island_matrix[i, j] == island_index:
                island_size += 1
    return island_size


def find_islands(f_image):
    """
    Identify connected components (islands) in a binary image.
    Uses a flood-fill-like algorithm to label connected regions.
    
    Args:
        f_image: Binary image (2D array with 0s and 1s)
    
    Returns:
        Tuple of (island_matrix, island_list)
        - island_matrix: Matrix with island labels
        - island_list: Array of [island_index, island_size] sorted by size
    """
    island_matrix = np.zeros(f_image.shape)
    island_index = 1
    
    for i in range(island_matrix.shape[0]):
        prima_riga = (i == 0)
        
        for j in range(island_matrix.shape[1]):
            prima_colonna = (j == 0)
            ultima_colonna = (j + 1 == island_matrix.shape[1])
            
            if f_image[i, j] == 1:  # Found a pixel that should be labeled
                island_array = np.zeros(4)
                
                if prima_riga:
                    island_array[0] = 0
                    island_array[1] = 0
                    island_array[2] = 0
                    island_array[3] = 0 if prima_colonna else island_matrix[i, j - 1]
                else:
                    if prima_colonna:
                        island_array[0] = 0
                        island_array[1] = island_matrix[i - 1, j]
                        island_array[2] = island_matrix[i - 1, j + 1]
                        island_array[3] = 0
                    elif ultima_colonna:
                        island_array[0] = island_matrix[i - 1, j - 1]
                        island_array[1] = island_matrix[i - 1, j]
                        island_array[2] = 0
                        island_array[3] = island_matrix[i, j - 1]
                    else:
                        island_array[0] = island_matrix[i - 1, j - 1]
                        island_array[1] = island_matrix[i - 1, j]
                        island_array[2] = island_matrix[i - 1, j + 1]
                        island_array[3] = island_matrix[i, j - 1]
                
                # Remove zeros (non-island neighbors)
                island_array = island_array[island_array != 0]
                island_array = np.unique(island_array)
                caso = len(island_array)
                
                if caso == 0:  # No neighboring islands - create new island
                    island_matrix[i, j] = island_index
                    island_index += 1
                elif caso == 1:  # One neighboring island - join it
                    island_matrix[i, j] = island_array[0]
                elif caso == 2:  # Two islands nearby - merge them
                    good_island = island_array[0]
                    island_matrix[i, j] = good_island
                    bad_island = island_array[1]
                    
                    # Merge the two islands
                    for y in range(i):
                        for x in range(island_matrix.shape[1]):
                            if island_matrix[y, x] == bad_island:
                                island_matrix[y, x] = good_island
                    
                    for x in range(j):
                        if island_matrix[i, x] == bad_island:
                            island_matrix[i, x] = good_island
    
    # Create sorted list of islands by size
    unique_islands = np.unique(island_matrix[:, :])[1:]  # Skip 0
    island_list = np.zeros((len(unique_islands), 2))
    
    for q in range(len(unique_islands)):
        island_size = find_island_size(island_matrix, unique_islands[q])
        island_list[q][0] = unique_islands[q]
        island_list[q][1] = island_size
    
    island_list = island_list[island_list[:, 1].argsort()]  # Sort by size
    
    return island_matrix, island_list


def erase_small_islands(image, clas, min_island_size):
    """
    Remove small disconnected regions (islands) for a specific class.
    
    Args:
        image: Segmentation image of shape (height, width, n_classes)
        clas: Class index to process
        min_island_size: Minimum size threshold - islands smaller than this are removed
    
    Returns:
        Image with small islands removed
    """
    island_matrix, island_list = find_islands(image[:, :, clas])
    
    # Erase all islands except the largest one, if they're below threshold
    for k in range(island_list.shape[0] - 1):  # Skip the largest island
        if island_list[k][1] < min_island_size:
            for i in range(island_matrix.shape[0]):
                for j in range(island_matrix.shape[1]):
                    if island_matrix[i][j] == island_list[k][0]:
                        image[i][j][clas] = 0  # Erase the island
                        
                        # Replace with background or head class
                        if clas != 6:
                            image[i][j][6] = 1  # Class "head" replaces the island
                        else:
                            image[i][j][0] = 1  # Class "background" replaces head island
    
    return image


def postprocess_predictions(predictions, min_island_size=12, apply_filtering=True, 
                           filter_size=3, num_classes=7):
    """
    Complete postprocessing pipeline for predictions.
    
    Args:
        predictions: Raw model predictions
        min_island_size: Minimum island size threshold
        apply_filtering: Whether to apply median filtering
        filter_size: Size of median filter
        num_classes: Number of classes
    
    Returns:
        Dictionary with different postprocessing results
    """
    results = {}
    
    # Step 1: Binarize predictions
    predictions_binary = binarize_predictions(predictions)
    results['binary'] = predictions_binary
    
    # Step 2: Apply median filtering (optional)
    if apply_filtering:
        predictions_filtered = apply_median_filter_to_predictions(
            predictions_binary, 
            filter_size
        )
        results['filtered'] = predictions_filtered
    else:
        predictions_filtered = predictions_binary
    
    # Step 3: Remove small islands from binary predictions
    predictions_islands = predictions_binary.copy()
    for imm in range(predictions_islands.shape[0]):
        for clas in range(num_classes):
            predictions_islands[imm, :, :, :] = erase_small_islands(
                predictions_islands[imm, :, :, :], 
                clas, 
                min_island_size
            )
    results['islands'] = predictions_islands
    
    # Step 4: Remove small islands from filtered predictions
    if apply_filtering:
        predictions_filtered_then_islands = predictions_filtered.copy()
        for imm in range(predictions_filtered_then_islands.shape[0]):
            for clas in range(num_classes):
                predictions_filtered_then_islands[imm, :, :, :] = erase_small_islands(
                    predictions_filtered_then_islands[imm, :, :, :], 
                    clas, 
                    min_island_size
                )
        results['filtered_islands'] = predictions_filtered_then_islands
    
    return results
