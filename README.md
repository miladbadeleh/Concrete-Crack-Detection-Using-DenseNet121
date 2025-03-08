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
The DenseNet121 model was trained for **10 epochs** on the custom dataset, achieving excellent performance across training, validation, and test sets. Below are the detailed results:

### Training Performance
- **Training Accuracy**: The model achieved a training accuracy of **99.77%** by the 10th epoch, with a training loss of **0.0089**.
- **Validation Accuracy**: The validation accuracy reached **99.81%** by the 10th epoch, with a validation loss of **0.0063**.
- **Test Accuracy**: The model achieved a test accuracy of **99.86%**, with a test loss of **0.0038**.

### Performance Over Epochs
The model showed consistent improvement over the training epochs:
- **Epoch 1**: Training Accuracy = 99.05%, Validation Accuracy = 99.71%
- **Epoch 2**: Training Accuracy = 99.59%, Validation Accuracy = 99.78%
- **Epoch 3**: Training Accuracy = 99.69%, Validation Accuracy = 99.86%
- **Epoch 4**: Training Accuracy = 99.67%, Validation Accuracy = 99.75%
- **Epoch 5**: Training Accuracy = 99.73%, Validation Accuracy = 99.85%
- **Epoch 6**: Training Accuracy = 99.76%, Validation Accuracy = 99.84%
- **Epoch 7**: Training Accuracy = 99.78%, Validation Accuracy = 99.83%
- **Epoch 8**: Training Accuracy = 99.79%, Validation Accuracy = 99.84%
- **Epoch 9**: Training Accuracy = 99.69%, Validation Accuracy = 99.89%
- **Epoch 10**: Training Accuracy = 99.77%, Validation Accuracy = 99.81%

### Key Observations
- The model achieved **high accuracy** (above 99%) on both the training and validation sets, indicating excellent learning and generalization capabilities.
- The **test accuracy** of **99.86%** demonstrates that the model performs well on unseen data, confirming its robustness.
- The **training and validation losses** consistently decreased over epochs, indicating stable and effective training.

### Final Metrics
| Metric          | Value       |
|-----------------|-------------|
| **Train Loss**  | 0.0089      |
| **Train Acc**   | 99.77%      |
| **Val Loss**    | 0.0063      |
| **Val Acc**     | 99.81%      |
| **Test Loss**   | 0.0038      |
| **Test Acc**    | 99.86%      |

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
