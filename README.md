# PetVision-UNet-Joint-Pixel-Level-Segmentation-and-Category-Recognition
MultiPet-Net is a multi-task deep learning framework based on a modified U-Net architecture. Designed to perform simultaneous pixel-level semantic segmentation and fine-grained species classification on the Oxford-IIT Pet Dataset, the network utilizes a shared convolutional encoder with dual task-specific output heads.


==> Key Features

1) Multi-Task Architecture: Combines semantic segmentation and 37-class image classification into a single unified model.

2) Shared Encoder Representation: Features a standard U-Net downsampling path that extracts rich feature maps for both downstream tasks, reducing total parameter redundancy.

*** Dual Output Heads:

3) Segmentation Decoder: Transpose-convolutional decoder with skip connections for precise boundary localization.

4) Classification Head: Adaptive average pooling and dense layers mapping deep features to 37 pet breed classes.

5) Lightweight Weights: State dict checkpointing (.pth) for fast inference and low storage footprint.

Model Architecture:

                   ┌──> [ Transpose Conv Decoder ] ──> Segmentation Mask (1x128x128)
[ Input Image ] ──>│ [ Conv Blocks + Pooling ]
                   └──> [ Adaptive AvgPool + FC ]   ──> Breed Class (37 Classes)

***Quick Start & Inference

Prerequisites:

a) PyTorch

b) torchvision

c) Pillow

d) matplotlib

***Running Inference:

Python
import torch
from model import MultiTaskUNet, process_and_show_image

# Path to sample image and saved weights
IMAGE_PATH = "path/to/pet_image.jpg"
WEIGHTS_PATH = "multitask_unet_pets.pth"

# Run prediction
process_and_show_image(IMAGE_PATH, WEIGHTS_PATH)
Results & Visualization
The model outputs both a binary segmentation mask isolating the pet from the background and a classification label predicting the exact pet breed index.
