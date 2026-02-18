The weights for the denoising model: $FRESTORMER$, can be access via this [drive file](https://drive.google.com/file/d/1665cUWqJj2wgo6r5RVN6hW8ixgFMGJmw/view?usp=sharing)

The **denoise_iterative** contains implementations and example scripts for a two-stage image denoising pipeline: *corrupted-pixel detection* followed by *noisy-pixel replacement/reconstruction*.

**Overview**
*Stage 1 — Detection:*

*AMF.py* — Adaptive Median Filter implementation for detecting corrupted colour pixels.

*AWMF.py* — Adaptive Weighted Mean Filter implementation for detecting corrupted colour pixels.

*Stage 2 — Reconstruction:*

*optimizer.py* — Conjugate-gradient–based optimizer used to solve the model objective for noisy-pixel denoising.

Examples:

*denoising_amf_iterative.ipynb* — Jupyter notebook demonstrating a sample workflow using the AMF implementation.

*denoising_awmf_iterative.ipynb* — Jupyter notebook demonstrating a sample workflow using the AWMF implementation.

If you find this repository useful, please cite:

Author(s), "Paper Title," *Denoising colour images corrupted by salt-and-pepper noise: weighted mean filter with iterative approach and deep learning modelling*, 2026.  
DOI: [10.1109/ACCESS.2026.3661859](https://doi.org/10.1109/ACCESS.2026.3661859)


