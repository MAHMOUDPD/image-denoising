
import numpy as np
from scipy.ndimage import median_filter
from skimage import io, color, util
from numba import njit, types, int64, prange
from numba.typed import List
import timeit

@njit
def AdapWeightedMeanF(img, max_window):
    """
    Phase 1: Here we use this function to perform Noise detection using adaptive weighted mean filter
    Input: img is an image
           max_wind is the maximum window size
    Outputs: noisy_mask and filtered, which are mask indixing set and set of weighted mean pixels    
    """
    H, W, C = img.shape
    noisy_mask = np.zeros((H, W, C), dtype=np.bool_)
    filtered = img.copy()
    
    for c in prange(C):
        channel = img[:, :, c]
        for i in prange(H):
            for j in prange(W):
                if channel[i, j] not in (0, 255):
                    continue
                
                Sminz = Smaxz = Smeanz = -1
                window_size = 3
                Sminzprev = 0
                Smaxzprev = 0 
                while window_size <= max_window:
                    # Extract current window
                    h_start = max(0, i - window_size // 2)
                    h_end = min(H, i + window_size // 2 + 1)
                    w_start = max(0, j - window_size // 2)
                    w_end = min(W, j + window_size // 2 + 1)
                    
                    window = channel[h_start:h_end, w_start:w_end]
                    Sminz = np.min(window)
                    Smeanz = get_weightedmean(window, Sminz, Smaxz) # np.median(window)
                    Smaxz = np.max(window)
        # Check stopping condition
                    if window_size > 3:
                        if Sminz == Sminzprev and Smaxz == Smaxzprev and Smeanz != -1:
                            break
                    Sminzprev = Sminz
                    Smaxzprev = Smaxz
                    window_size += 2

                if Sminz < channel[i, j] < Smaxz:
                    continue  # Keep original pixel value
                else:
                    filtered[i, j, c] = Smeanz
                    noisy_mask[i, j, c] = True

    return noisy_mask, filtered

@njit
def get_weightedmean(window, Sminz, Smaxz):
    # Handle scalar or empty window cases
    if window.size==0 or window.ndim==0:
        return -1.0
    # Initialize weights matrix
    weights=np.zeros_like(window)
    valid_count=0
    # Process each element in the window
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            val=window[i, j]
            if Sminz < val < Smaxz:
                weights[i, j]=1
                valid_count+=1
    # Check if any valid pixels were found
    if valid_count==0:
        return -1.0
    # Calculate weighted mean
    total_sum=0.0
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            total_sum+=weights[i, j]*window[i, j]
    
    return total_sum/valid_count