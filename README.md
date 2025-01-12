# Emotion Recognition using Vision Transformers (ViT) on FER2013

This project implements a Vision Transformer (ViT) model for emotion recognition on the FER2013 dataset. The model is trained using PyTorch and achieves a **test accuracy of 70.34%**. The project includes dynamic learning rate scheduling, data augmentation, and early stopping to optimize performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Repository Contents](#repository-contents)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project explores the use of Vision Transformers (ViT) for emotion recognition on the FER2013 dataset. The ViT model is fine-tuned using PyTorch, and dynamic learning rate scheduling is applied to optimize training. Data augmentation techniques are also used to improve generalization.

## Dataset
The FER2013 dataset consists of 35,887 grayscale images of faces, each labeled with one of seven emotions:
- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

The dataset is split into training (28,709 images) and test (7,178 images) sets.

## Model Architecture
The model is based on the Vision Transformer (ViT) architecture, specifically the `google/vit-base-patch16-224` model. The model is fine-tuned for emotion recognition by replacing the final classification head with a 7-class output layer.

## Training
The model is trained using the following techniques:
- **Dynamic Learning Rate Scheduling**: The learning rate is adjusted using `ReduceLROnPlateau` to improve convergence.
- **Data Augmentation**: Random horizontal flips, rotations, translations, and color jittering are applied to the training data.
- **Early Stopping**: Training stops if the validation loss does not improve for 3 consecutive epochs.

## Repository Contents
- **`app.ipynb`**: Jupyter notebook for running an app that uses the trained model for emotion recognition.
- **`finetuning.ipynb`**: Jupyter notebook for fine-tuning the Vision Transformer (ViT) model on the FER2013 dataset.
- **`model/vit_fer2013.pth`**: Trained model weights.

## Usage
1. **Fine-Tuning the Model**:
   - Open the `finetuning.ipynb` notebook and follow the instructions to fine-tune the ViT model on the FER2013 dataset.

2. **Running the App**:
   - Open the `app.ipynb` notebook and use the trained model to predict emotions from images.

## Results
The fine-tuned model achieves a **test accuracy of 70.34%** on the FER2013 dataset.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

