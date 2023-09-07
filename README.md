# cGan for Handwritten Digits Generation

## Introduction

This repository contains a Conditional Generative Adversarial Network (cGAN) that generates handwritten digits. The cGAN network consists of a Generator and a Discriminator, both implemented using TensorFlow's Keras API.

## Features

- **Data Set**: Uses the MNIST data set for handwritten digits.
- **Discriminator**: Uses Leaky ReLU activation, Binary Cross-Entropy loss, and Adam optimizer.
- **Generator**: Utilizes dense layers, batch normalization, and Leaky ReLU activation functions.
- **Hyperparameters**:  
    - Batch size \(= 128\)  
    - Epochs \(= 1000\)  
    - Latent dimensions \(= 100\)

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Code Structure

### Hyperparameters

batch_size = 128
epochs = 1000
latent_dim = 100

### Generator

- Takes a noise vector (shape: \(100, 1\)) and a label (shape: \(10, 1\)).
- Outputs a generated image (shape: \(784, 1\)).

### Discriminator

- Takes a real/generated image (shape: \(784, 1\)) and a label (shape: \(10, 1\)).
- Outputs a validity score between 0 and 1.

### Training Loop

The training loop involves alternating between training the Discriminator and the Generator.

### Digit Generation Script

Load the pre-trained generator model and use it to generate and plot handwritten digits.


