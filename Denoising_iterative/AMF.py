import numpy as np
from scipy.ndimage import median_filter
from skimage import io, color, util
from numba import njit, types, int64, prange
from numba.typed import List
import timeit

@njit
def AdapMedianFilter(img, max_window):
    """
    Phase 1: Here we use this function to perform Noise detection using adaptive median filter
    Input: img is an image
           max_wind is the maximum window size
    Outputs: noisy_mask and filtered, which arße mask indixing set and set of median pixels    
    """
    noisy_mask = np.zeros(img.shape, dtype=np.bool_)
    filtered = img.copy()
    
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if channel[i, j] != 0 and channel[i, j] != 255:
                    continue
                window_size = 3
                while window_size <= max_window // 2:
                    h_start = max(0, i - window_size // 2)
                    h_end = min(channel.shape[0], i + window_size // 2 + 1)
                    w_start = max(0, j - window_size // 2)
                    w_end = min(channel.shape[1], j + window_size // 2 + 1)
                    
                    window = channel[h_start:h_end, w_start:w_end]
                    w_min = window.min()
                    w_med = np.median(window)
                    w_max = window.max()

                    if w_min < w_med < w_max:
                        if w_min < channel[i, j] < w_max:
                            break
                        else:
                            filtered[i, j, c] = w_med
                            noisy_mask[i, j, c] = True
                            break
                    window_size += 2
                else:
                    filtered[i, j, c] = w_med
    return noisy_mask, filtered
