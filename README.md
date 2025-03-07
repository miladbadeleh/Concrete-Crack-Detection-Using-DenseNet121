# DenseNet121 Image Classification

This repository contains a PyTorch implementation of the DenseNet121 architecture for image classification. The model is trained on a custom dataset and evaluated on validation and test sets. Below is a detailed explanation of the code, architecture, and workflow.

## Table of Contents
1. [Introduction](#introduction)
2. [DenseNet Architecture](#densenet-architecture)
3. [Code Overview](#code-overview)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Usage](#usage)
7. [Dependencies](#dependencies)

---

## Introduction
This project uses the DenseNet121 architecture, a deep convolutional neural network (CNN), to classify images into predefined categories. DenseNet (Densely Connected Convolutional Networks) is known for its efficient use of parameters and its ability to alleviate the vanishing gradient problem by connecting each layer to every other layer in a feed-forward fashion.

The code is implemented using PyTorch, a popular deep learning framework, and is designed to work with custom datasets. The model is trained, validated, and tested on image datasets, and the trained model is saved for future use.

---

## DenseNet Architecture
DenseNet121 is a variant of the DenseNet architecture, which consists of 121 layers. The key features of DenseNet are:

- **Dense Blocks**: Each layer in a dense block receives feature maps from all preceding layers and passes its own feature maps to all subsequent layers. This encourages feature reuse and improves gradient flow.
- **Transition Layers**: These layers are used between dense blocks to reduce the spatial dimensions of the feature maps (e.g., using pooling).
- **Bottleneck Layers**: To improve computational efficiency, bottleneck layers (1x1 convolutions) are used to reduce the number of input feature maps before applying 3x3 convolutions.
- **Growth Rate**: This hyperparameter controls how many new feature maps are added to the network in each layer.

The final layer of DenseNet121 is a fully connected layer (`classifier`) that maps the extracted features to the number of output classes.

---

## Code Overview
The code is structured as follows:

1. **Data Loading and Preprocessing**:
   - Images are resized to 224x224 and normalized using ImageNet mean and standard deviation.
   - The dataset is split into training, validation, and test sets using `torchvision.datasets.ImageFolder`.

2. **Model Definition**:
   - The DenseNet121 model is loaded with pretrained weights using `torchvision.models.densenet121(pretrained=True)`.
   - The final fully connected layer (`classifier`) is modified to match the number of classes in the dataset.

3. **Training**:
   - The model is trained using the Adam optimizer and CrossEntropyLoss.
   - Training and validation loops are implemented with progress bars using `tqdm`.
   - Training statistics (loss and accuracy) are printed for each epoch.

4. **Evaluation**:
   - The model is evaluated on the test set, and test statistics (loss and accuracy) are printed.

5. **Model Saving**:
   - The trained model is saved to a file (`densenet121_model.pth`) for future use.

---

## Training and Evaluation
### Training
- The model is trained for 10 epochs by default.
- Training and validation losses are monitored to ensure the model is learning effectively.
- The Adam optimizer is used with a learning rate of 0.001.

### Evaluation
- The model is evaluated on a separate test set to measure its generalization performance.
- Test accuracy and loss are reported.

---

## Results
After training, the model achieves the following results:
- **Training Accuracy**: [Insert training accuracy]
- **Validation Accuracy**: [Insert validation accuracy]
- **Test Accuracy**: [Insert test accuracy]

---

## Usage
To use this code, follow these steps:

1. **Set Up the Environment**:
   - Install the required dependencies (see [Dependencies](#dependencies)).
   - Mount Google Drive (if using Google Colab) and organize your dataset into `train`, `validation`, and `test` folders.

2. **Run the Code**:
   - Execute the notebook or script to train the model.
   - The trained model will be saved to `/content/drive/MyDrive/densenet121_model.pth`.

3. **Test the Model**:
   - Use the `test` function to evaluate the model on the test set.

---

## Dependencies
The following Python libraries are required to run this code:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `PIL`

Install the dependencies using:
```bash
pip install torch torchvision tqdm matplotlib pillow
