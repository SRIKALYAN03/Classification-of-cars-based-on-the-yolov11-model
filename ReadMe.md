# YOLOv11 Classification Model Training and Deployment

This repository contains a Python implementation for organizing datasets, training a classification model using YOLOv11, and deploying it for real-world applications. The project is designed to handle custom datasets and provides a complete pipeline from dataset preparation to model saving.

## Task Description

The primary objective of this project is to classify car images based on their models and variants using a fine-tuned YOLOv11 classification model. The workflow includes:

1. **Organizing raw car image datasets** into structured folders for training, validation, and testing.
2. **Fine-tuning a pre-trained YOLOv11 model** on the prepared dataset to classify images into specific car models and variants.
3. **Saving the trained model** for deployment and inference tasks.
4. **Performing predictions** on new images to identify the car model and variant accurately.

This task aims to create a robust and scalable classification system for car images, which can be used for various applications, including inventory management, automated inspections, and customer-facing platforms.

## Features

- **Dataset Preparation**: Automatically organizes raw images into `train`, `val`, and `test` splits.
- **Model Training**: Fine-tunes a YOLOv11 classification model on the prepared dataset.
- **Model Saving**: Saves the trained model in `.pt` format for future use.
- **Model Deployment**: Provides tools to load and preprocess data for model inference.

## Project Structure

- `organize_dataset`: Script to split the dataset into `train`, `val`, and `test` directories.
- `prepare_dataset`: Function to apply transformations and load datasets into PyTorch DataLoader objects.
- `train_model`: Fine-tunes the YOLOv11 classification model.
- `save_model`: Saves the trained model checkpoint.
- `preprocess_image`: Prepares individual images for model inference.
- `model_inference`: Loads the trained model for predictions.

## Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- ultralytics
- scikit-learn
- tqdm
- PIL (Pillow)

Install the required packages using:
```bash
pip install torch torchvision ultralytics scikit-learn tqdm pillow
