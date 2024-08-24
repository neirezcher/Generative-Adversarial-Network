# Deep Learning with PyTorch: Generative Adversarial Network (GAN)

## Project Overview

This project implements a **Generative Adversarial Network (GAN)** using **PyTorch**, a widely-used deep learning framework. The GAN architecture consists of two competing neural networks: the **Generator** and the **Discriminator**. The Generator aims to create realistic synthetic data that mimics a given dataset, while the Discriminator's role is to distinguish between real and generated data. This adversarial process drives both networks to improve their performance over time.

## Key Features

- **Data Generation**: The Generator learns to produce high-quality images resembling those from the training dataset, specifically focusing on generating handwritten digits from the **MNIST** dataset.
  
- **Discriminator Training**: The Discriminator is trained to effectively classify real and generated images, enhancing its accuracy through adversarial training.

- **Loss Function**: The project utilizes **Binary Cross-Entropy Loss** to measure the performance of both networks, enabling effective learning via backpropagation.

- **Optimizers**: **Adam optimizers** are employed for both the Generator and the Discriminator, ensuring stable and efficient training dynamics.

## Dataset

The model is trained on the **MNIST dataset**, which consists of 70,000 images of handwritten digits (0-9) in grayscale format, each with a resolution of 28x28 pixels. The dataset is divided into training and testing sets, with the training set used for model training and the testing set for evaluation.

## Results

After training, the model generates synthetic images that can be visually evaluated using **Matplotlib**. This allows for a comparison between real and generated images, showcasing the effectiveness of the GAN model.
