'''
Written by: Mahmoud M. Yahaya
Contact address: mahmoudpd@gmail.com
Date of last update: 19/08/2024
'''

import numpy as np
from scipy.ndimage import median_filter
from skimage import io, color, util
from numba import njit, types, int64, prange
from numba.typed import List
import timeit


# Edge-preserving potential function and its derivative
@njit
def phi(t, alpha):
    return np.sqrt(t**2+alpha)

@njit
def dPhi(t, alpha):
    return t/np.sqrt(t**2+alpha)

@njit(parallel=True)
def get_grad(img, u, y, noisy_mask, index_map, beta, alpha):
    H, W, C=img.shape
    grad=np.zeros(len(u), dtype=np.float64)  # here we used float64 for consistency purpose
    # Offsets for 4-connected neighbors
    offsets=[(0, 1),(1, 0),(0, -1),(-1, 0)]
    # Process each channel in parallel
    for c in prange(C):
        for i in prange(H):
            for j in prange(W):
                if not noisy_mask[i,j,c]:
                    continue  
                idx=index_map[i,j,c]
                sum_deriv=0.0
                # process neighbors
                for di, dj in offsets:
                    ni, nj = i+di, j+dj
                    if 0<=ni<H and 0<=nj<W:
                        # here we check if neighbor is noisy
                        if noisy_mask[ni, nj, c]:
                            nbr_idx=index_map[ni, nj, c]
                            diff = u[idx] - u[nbr_idx]
                            sum_deriv+=dPhi(diff, alpha)
                        #then the Neighbor is clean
                        else:
                            diff = u[idx] - y[ni, nj, c]
                            sum_deriv+=2*dPhi(diff, alpha)
                grad[idx] = beta*sum_deriv                
    return grad

@njit
def getindex_map(noisy_mask):
    H, W, C=noisy_mask.shape
    index_map=np.full((H, W, C), -1, dtype=np.int64)
    idx = 0
    for i in prange(H):
        for j in prange(W):
            for c in prange(C):
                if noisy_mask[i, j, c]:
                    index_map[i, j, c]=idx
                    idx+=1
    return index_map, idx


import matplotlib.pyplot as plt

def plot_images(img1, img2, denoised_image, titles=None):
    """
    Plot three images side-by-side in a single row with three columns.
    
    Args:
        img1: First input image (numpy array or PIL image)
        img2: Second input image (numpy array or PIL image)
        denoised_image: Denoised result image (numpy array or PIL image)
        titles: Optional list of titles for each image
    """
    if titles is None:
        titles = ['Image 1', 'Image 2', 'Denoised Image']
    
    plt.figure(figsize=(15, 5))
    
    # Plot first image
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title(titles[0])
    plt.axis('off')
    
    # Plot second image
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title(titles[1])
    plt.axis('off')
    
    # Plot denoised image
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image)
    plt.title(titles[2])
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
