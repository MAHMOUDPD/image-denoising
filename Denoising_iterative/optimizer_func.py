import numpy as np
from scipy.ndimage import median_filter
from skimage import io, color, util
from numba import njit, types, int64, prange
from numba.typed import List
import timeit
from utils_funcs import *

@njit
def DenoisePRCG(img,noisy_mask,filtered_pix, beta, tol, max_iter, alpha):
    '''
     DenoisePRCG is a Polak–Ribière based conjugate gradient method that used an adaptive 
     step-size computing step
    
    :param img: is the input image
    :param filter_func: is filter function, here used t can be AMF or AWMF
    :param beta: is a parameter used in the denoising functional, here we set to 2
    :param tol: is a tolerance value
    :param max_iter: is a choicing maximum number of iterations
    :param alpha: is a parameter in the edge preserving functional
    '''
    # Phase 1: Noise detection
    # noisy_mask, filtered_pix=filter_func(img, max_window)
    y = filtered_pix.astype(np.float32)/255.0
    # Create index map for noisy pixels
    index_mapped, num_noisy=getindex_map(noisy_mask)
    x = np.zeros(num_noisy, dtype=np.float64) # here we adopt float64 for optimization variables to prevent type conflicts
    for i in range(noisy_mask.shape[0]):
        for j in range(noisy_mask.shape[1]):
            for c in range(noisy_mask.shape[2]):
                if noisy_mask[i, j, c]:
                    idx=index_mapped[i, j, c]
                    x[idx]=np.float64(y[i, j, c])  # float64 convertion
    g_prev=np.zeros(num_noisy, dtype=np.float64)
    d_prev=np.zeros(num_noisy, dtype=np.float64)
    converged = False
    for t in range(max_iter):
        # Compute gradient - returns float64
        g = get_grad(img,x,y,noisy_mask,index_mapped,beta,alpha)
        # Compute beta (Polak-Ribiere)
        if t==0:
            beta_param = 0.0
        else:
            numerator = np.sum(g*(g-g_prev))
            denominator = np.sum(g_prev*g_prev)
            beta_param = max(0, numerator/(denominator+1e-8))
        # Update search direction
        d = -g+beta_param*d_prev
        # Compute step size
        numerator = np.sum(g*d)
        denominator = np.sum(d*d)
        step = -0.1*numerator/(denominator+1e-8)
        # Update solution
        x_new = x+step*d
        # Check convergence
        diff_norm = np.linalg.norm(x_new - x)
        base_norm = np.linalg.norm(x)
        rel_change = diff_norm/(base_norm+1e-8)
        if rel_change < tol:
            converged = True
            x = x_new
            break
        # Update for next iteration
        d_prev = d
        g_prev = g
        x = x_new
    # Reconstruct final image (convert back to float32)
    denoised=y.copy()
    for i in range(noisy_mask.shape[0]):
        for j in range(noisy_mask.shape[1]):
            for c in range(noisy_mask.shape[2]):
                if noisy_mask[i, j, c]:
                    idx=index_mapped[i, j, c]
                    denoised[i, j, c]=np.float32(x[idx])  # Convert back to float32
    return (denoised*255).astype(np.uint8)