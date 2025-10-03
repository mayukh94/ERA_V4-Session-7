# CIFAR-10 Classification with Modular PyTorch Code

This repository provides a modular implementation for training a deep learning model on the CIFAR-10 dataset using PyTorch. The code is organized into separate files for the model definition (`model.py`) and the training pipeline (`train.py`), following best practices for maintainability and clarity.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Receptive Field (RF) Calculation](#receptive-field-rf-calculation)
- [References](#references)

---

## Project Structure

```
.
├── app.py
├── main.py
├── model.py         # Model architecture (modular)
├── train.py         # Training and testing loops (modular)
├── static/
├── templates/
├── README.md
└── ...
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd python-code
   ```

2. **Install dependencies:**
   - It's recommended to use a virtual environment.
   - Install required packages:
     ```bash
     pip install torch torchvision albumentations matplotlib tqdm torchsummary
     ```

---

## How to Run

1. **Train the Model:**
   - Run the training script:
     ```bash
     python train.py
     ```
   - This will:
     - Load the model from `model.py`
     - Prepare CIFAR-10 data with augmentations
     - Train and evaluate the model, printing progress and results

2. **(Optional) Model Summary:**
   - To view the model summary, ensure `torchsummary` is installed and use:
     ```python
     from torchsummary import summary
     from model import CIFARNet
     model = CIFARNet().to('cuda' if torch.cuda.is_available() else 'cpu')
     summary(model, input_size=(3, 32, 32))
     ```

---

## Model Architecture

The model is a custom CNN for CIFAR-10, featuring:

- **Initial Conv Layer:** Standard 3x3 convolution
- **Depthwise Separable Convolution:** Efficient feature extraction
- **Dilated Convolution:** Enlarges receptive field without increasing parameters
- **Strided Convolutions:** For downsampling
- **Global Average Pooling:** Reduces each feature map to a single value
- **Fully Connected Layer:** Outputs class probabilities

**Layer-wise breakdown:**
| Layer                | Output Channels | Kernel/Stride/Pad | Special         |
|----------------------|----------------|-------------------|-----------------|
| Conv2d + BN + ReLU   | 32             | 3x3/1/1           |                 |
| DepthwiseSeparable   | 64             | 3x3/2/1           | Depthwise       |
| Conv2d (dilated)     | 96             | 3x3/1/2, dil=2    | Dilated         |
| Conv2d + BN + ReLU   | 128            | 3x3/2/1           |                 |
| GAP                  | 128            | -                 |                 |
| Linear               | 10             | -                 | Output classes  |

---

## Receptive Field (RF) Calculation

The receptive field (RF) is the region in the input image that affects a particular output value. For this architecture:

- **Conv1:** 3x3, stride 1 → RF = 3
- **DepthwiseSeparable (stride 2):** 3x3, stride 2 → RF = 7
- **Dilated Conv (dilation 2):** 3x3, stride 1, dilation 2 → RF = 11
- **Conv4 (stride 2):** 3x3, stride 2 → RF = 19
- **GAP:** Covers the entire feature map

**Total RF at output:** 19x19 pixels (before GAP), meaning each output "sees" a 19x19 region of the input.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Albumentations](https://albumentations.ai/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Notes

- You can modify `model.py` to experiment with different architectures.
- Adjust hyperparameters in `train.py` as needed.
- For visualization and debugging, use the plotting code provided in the notebook or scripts.

---
