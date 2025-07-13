# ARW-JNBA

This repository contains a PyTorch implementation of an unsupervised method for segmenting foreground objects in underwater images.

## Requirements
* Python 3.8+
* PyTorch 1.9+
* Torchvision
* OpenCV-Python (`opencv-python`)
* NumPy

## Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/LLL-YUE/ARW-JNBA.git](https://github.com/LLL-YUE/ARW-JNBA.git)
    cd ARW-JNBA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Using conda
    conda create -n arw_usod python=3.8
    conda activate arw_usod
    ```

## Dataset Preparation

The model can be evaluated on several public datasets. Please download them from their official sources:

* **SUIM:** https://irvlab.cs.umn.edu/resources/suim-dataset
* **USOD10K:** https://github.com/LinHong-HIT/USOD10K
* **UFO-120:** https://irvlab.cs.umn.edu/resources/ufo-120-dataset
