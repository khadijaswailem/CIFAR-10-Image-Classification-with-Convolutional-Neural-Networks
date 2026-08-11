[README.md](https://github.com/user-attachments/files/30934572/README.md)
# CIFAR-10 Image Classification with Convolutional Neural Networks

This project builds, trains, and evaluates several Convolutional Neural Network architectures on the CIFAR-10 dataset using PyTorch. The goal was to explore how depth, width, pooling strategy, and regularization choices affect model performance on a small, low resolution image classification task.

## About the dataset

CIFAR-10 contains 60,000 color images across 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32x32 pixels with 3 color channels. The data was split into 45,000 training images, 5,000 validation images, and 10,000 test images.

## What this project covers

**Data exploration and preprocessing**
Class balance checks, sample visualizations, and per channel normalization (mean and standard deviation computed directly from the training set) before feeding images into the models.

**Four CNN architectures**
Four models were designed with increasing depth and width, each with a short justification for its design choices:

- **CNN1**: A simple 2 layer network (32 to 64 filters) with batch normalization and dropout, used as a baseline.
- **CNN2**: A 3 layer network (64 to 128 to 128 filters), used as the main model for later comparisons.
- **CNN3**: A 4 layer network with paired convolutions in a VGG style block structure (32 to 64 to 128 to 128 filters).
- **CNN4**: The deepest model at 5 layers (32 to 64 to 128 to 256 filters), combining paired convolutions with a wider final block and layered dropout.

All models use 3x3 convolution filters throughout, since the images are already small and larger filters were not expected to help.

**Max pooling vs average pooling**
CNN2 was retrained with average pooling instead of max pooling, and the two versions are compared side by side on training curves, validation loss, and test accuracy to see how each pooling method affects generalization.

**Architecture analysis**
A manual, layer by layer breakdown of CNN2's output shapes and parameter counts, working through the math behind each convolution and pooling operation.

**Model evaluation**
All four models are reloaded from saved weights and evaluated on training and test accuracy for a direct comparison.

**Hyperparameter tuning**
CNN2 was tuned further using a configurable version of the architecture, testing different learning rates, dropout rates, and the presence or absence of batch normalization.

**Reflection**
Closing notes on what the learned filters look like at different depths, how max and average pooling compare in practice, and how the number of filters affects the total parameter count.

## Tools and libraries

- PyTorch and torchvision for models, training, and data loading
- scikit-learn for classification reports and confusion matrices
- matplotlib and seaborn for visualizations
- Google Colab (with Google Drive used to store model checkpoints)

## Repository contents

- `Assignment_2_CNN_khadija_swailem.ipynb`: the full notebook, including data exploration, model definitions, training runs, evaluation, and written analysis after each experiment.

## Notes

This was built as a coursework assignment, so the notebook includes written commentary and reflection alongside the code, explaining the reasoning behind each design decision and what was learned from each experiment. Training was limited to 5 to 8 epochs per model due to Colab runtime constraints, so results reflect early stage training behavior rather than fully converged models.
