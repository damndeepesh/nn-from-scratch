# Neural Network from Scratch

This project implements a neural network from scratch using only NumPy, without relying on machine learning libraries like TensorFlow or PyTorch. It includes a Gradio UI for interactive experimentation.

## Features

### Core Neural Network Components

- **Matrix Operations**: Implemented using NumPy arrays
- **Feed-Forward Architecture**: Customizable multi-layer perceptron
- **Backpropagation Algorithm**: Complete gradient descent optimization

### Advanced Features

- **Multiple Activation Functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Regularization Techniques**: L1, L2 regularization
- **Weight Initialization Strategies**: Xavier/Glorot, He, Zero, Random Normal
- **Batch Normalization**: Normalize layer inputs for faster training
- **Dropout**: Prevent overfitting
- **Learning Rate Scheduling**: Constant, Step Decay, Exponential Decay, Time-Based Decay
- **Early Stopping**: Prevent overfitting by monitoring validation loss

### Visualization and Datasets

- **Training Visualization**: Loss and accuracy plots
- **Decision Boundary Visualization**: For classification problems
- **Synthetic Datasets**: XOR, Circle, Spiral, Regression problems

## Getting Started

### Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

Start the Gradio UI with:

```bash
python app.py
```

### Usage Guide

1. **Generate Dataset**: Choose from various synthetic datasets with customizable parameters
2. **Create Model**: Design your neural network architecture with custom layers and hyperparameters
3. **Train Model**: Train your model with various optimization settings
4. **Evaluate & Visualize**: Analyze model performance and visualize predictions

## Implementation Details

### Layer Types

- **Dense**: Fully connected layer with customizable activation functions
- **Dropout**: Regularization layer that randomly sets inputs to zero
- **BatchNormalization**: Normalizes layer inputs for more stable training

### Mathematical Foundation

The implementation follows the standard neural network mathematics:

- **Forward Pass**: z = Wx + b, a = activation(z)
- **Backward Pass**: Compute gradients using the chain rule
- **Weight Updates**: w = w - learning_rate * gradient

## Examples

### XOR Problem

The XOR problem is a classic non-linearly separable problem that requires at least one hidden layer to solve:

1. Generate an XOR dataset
2. Create a model with one hidden layer (4 neurons) and ReLU activation
3. Train with binary cross-entropy loss
4. Visualize the decision boundary

### Classification Problems

For more complex classification:

1. Generate a spiral dataset with 3 classes
2. Create a model with two hidden layers (10, 10) with ReLU activation and dropout
3. Train with categorical cross-entropy loss
4. Evaluate accuracy and visualize decision boundaries

## License

This project is open source and available under the MIT License.