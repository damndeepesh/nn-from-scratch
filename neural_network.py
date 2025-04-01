import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import List, Tuple, Dict, Callable, Union, Optional


class Initializer:
    """
    Class for weight initialization strategies
    """
    @staticmethod
    def xavier(shape: Tuple[int, int]) -> np.ndarray:
        """
        Xavier/Glorot initialization - good for tanh/sigmoid activations
        """
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def he(shape: Tuple[int, int]) -> np.ndarray:
        """
        He initialization - good for ReLU activations
        """
        n_in, n_out = shape
        std = np.sqrt(2 / n_in)
        return np.random.normal(0, std, shape)
    
    @staticmethod
    def zero(shape: Tuple[int, int]) -> np.ndarray:
        """
        Zero initialization
        """
        return np.zeros(shape)
    
    @staticmethod
    def random_normal(shape: Tuple[int, int], mean: float = 0.0, std: float = 0.1) -> np.ndarray:
        """
        Random normal initialization
        """
        return np.random.normal(mean, std, shape)


class Activation:
    """
    Class for activation functions and their derivatives
    """
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent activation function
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh: 1 - tanh(x)^2
        """
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit activation function
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: 1 if x > 0 else 0
        """
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU activation function
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Derivative of Leaky ReLU
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function
        """
        # Shift x for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of softmax (simplified for cross-entropy loss)
        """
        # For softmax with cross-entropy, this is handled in the loss function
        return np.ones_like(x)


class Loss:
    """
    Class for loss functions and their derivatives
    """
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error loss function
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE with respect to y_pred
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Binary Cross-Entropy loss function
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Derivative of Binary Cross-Entropy with respect to y_pred
        """
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -((y_true / y_pred) - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Categorical Cross-Entropy loss function
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of Categorical Cross-Entropy with respect to y_pred
        For softmax activation, this simplifies to (y_pred - y_true)
        """
        return (y_pred - y_true) / y_true.shape[0]


class Regularizer:
    """
    Class for regularization techniques
    """
    @staticmethod
    def l1(weights: np.ndarray, lambda_: float) -> float:
        """
        L1 regularization term
        """
        return lambda_ * np.sum(np.abs(weights))
    
    @staticmethod
    def l1_derivative(weights: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Derivative of L1 regularization
        """
        return lambda_ * np.sign(weights)
    
    @staticmethod
    def l2(weights: np.ndarray, lambda_: float) -> float:
        """
        L2 regularization term
        """
        return 0.5 * lambda_ * np.sum(np.square(weights))
    
    @staticmethod
    def l2_derivative(weights: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Derivative of L2 regularization
        """
        return lambda_ * weights


class LearningRateScheduler:
    """
    Class for learning rate scheduling strategies
    """
    @staticmethod
    def constant(initial_lr: float, epoch: int) -> float:
        """
        Constant learning rate
        """
        return initial_lr
    
    @staticmethod
    def step_decay(initial_lr: float, epoch: int, drop_rate: float = 0.5, epochs_drop: int = 10) -> float:
        """
        Step decay learning rate
        """
        return initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr: float, epoch: int, decay_rate: float = 0.95) -> float:
        """
        Exponential decay learning rate
        """
        return initial_lr * np.power(decay_rate, epoch)
    
    @staticmethod
    def time_based_decay(initial_lr: float, epoch: int, decay_rate: float = 0.01) -> float:
        """
        Time-based decay learning rate
        """
        return initial_lr / (1 + decay_rate * epoch)


class Layer:
    """
    Base class for neural network layers
    """
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        """
        raise NotImplementedError
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer
        """
        raise NotImplementedError
    
    def get_params(self) -> Dict:
        """
        Get layer parameters for saving
        """
        return {}
    
    def set_params(self, params: Dict) -> None:
        """
        Set layer parameters from loaded data
        """
        pass


class Dense(Layer):
    """
    Fully connected layer implementation
    """
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'relu',
                 weight_init: str = 'he',
                 use_bias: bool = True,
                 regularization: Optional[Tuple[str, float]] = None):
        super().__init__()
        self.weights = self._initialize_weights(input_size, output_size, weight_init)
        self.bias = np.zeros((1, output_size)) if use_bias else None
        self.use_bias = use_bias
        
        # Set activation function and its derivative
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'leaky_relu':
            self.activation = Activation.leaky_relu
            self.activation_derivative = Activation.leaky_relu_derivative
        elif activation == 'softmax':
            self.activation = Activation.softmax
            self.activation_derivative = Activation.softmax_derivative
        elif activation == 'linear':
            self.activation = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Set regularization
        self.regularization = regularization
        if regularization:
            reg_type, lambda_ = regularization
            if reg_type == 'l1':
                self.reg_fn = lambda: Regularizer.l1(self.weights, lambda_)
                self.reg_derivative = lambda: Regularizer.l1_derivative(self.weights, lambda_)
            elif reg_type == 'l2':
                self.reg_fn = lambda: Regularizer.l2(self.weights, lambda_)
                self.reg_derivative = lambda: Regularizer.l2_derivative(self.weights, lambda_)
            else:
                raise ValueError(f"Unsupported regularization type: {reg_type}")
        else:
            self.reg_fn = lambda: 0
            self.reg_derivative = lambda: 0
    
    def _initialize_weights(self, input_size: int, output_size: int, method: str) -> np.ndarray:
        """
        Initialize weights based on the specified method
        """
        shape = (input_size, output_size)
        if method == 'xavier':
            return Initializer.xavier(shape)
        elif method == 'he':
            return Initializer.he(shape)
        elif method == 'zero':
            return Initializer.zero(shape)
        elif method == 'random':
            return Initializer.random_normal(shape)
        else:
            raise ValueError(f"Unsupported weight initialization method: {method}")
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = input @ weights + bias, a = activation(z)
        """
        self.input = input_data
        self.z = np.dot(self.input, self.weights)
        if self.use_bias:
            self.z += self.bias
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass: compute gradients and update weights
        """
        # If the output_gradient is coming from softmax + categorical_crossentropy,
        # we can skip the activation derivative calculation
        if self.activation_name == 'softmax':
            activation_gradient = output_gradient
        else:
            activation_gradient = output_gradient * self.activation_derivative(self.z)
        
        # Calculate weight gradients
        weights_gradient = np.dot(self.input.T, activation_gradient) + self.reg_derivative()
        
        # Calculate bias gradients if using bias
        if self.use_bias:
            bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
            self.bias -= learning_rate * bias_gradient
        
        # Calculate input gradients for the previous layer
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        # Update weights
        self.weights -= learning_rate * weights_gradient
        
        return input_gradient
    
    def get_params(self) -> Dict:
        """
        Get layer parameters for saving
        """
        params = {
            'type': 'Dense',
            'weights': self.weights.tolist(),
            'activation': self.activation_name,
            'use_bias': self.use_bias
        }
        if self.use_bias:
            params['bias'] = self.bias.tolist()
        if self.regularization:
            params['regularization'] = self.regularization
        return params
    
    def set_params(self, params: Dict) -> None:
        """
        Set layer parameters from loaded data
        """
        self.weights = np.array(params['weights'])
        if 'bias' in params and params['use_bias']:
            self.bias = np.array(params['bias'])
        self.use_bias = params['use_bias']
        self.activation_name = params['activation']
        if 'regularization' in params:
            self.regularization = params['regularization']


class Dropout(Layer):
    """
    Dropout layer for regularization
    """
    def __init__(self, rate: float = 0.2):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass with dropout during training
        """
        self.input = input_data
        
        if self.training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            # During inference, no dropout is applied
            self.output = input_data
            
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass with dropout mask
        """
        if self.training:
            return output_gradient * self.mask
        else:
            return output_gradient
    
    def get_params(self) -> Dict:
        """
        Get layer parameters for saving
        """
        return {
            'type': 'Dropout',
            'rate': self.rate
        }
    
    def set_params(self, params: Dict) -> None:
        """
        Set layer parameters from loaded data
        """
        self.rate = params['rate']


class BatchNormalization(Layer):
    """
    Batch Normalization layer
    """
    def __init__(self, input_shape: int, epsilon: float = 1e-8, momentum: float = 0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, input_shape))
        self.beta = np.zeros((1, input_shape))
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, input_shape))
        self.running_var = np.ones((1, input_shape))
        
        # Training mode flag
        self.training = True
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass with batch normalization
        """
        self.input = input_data
        
        if self.training:
            # Calculate batch statistics
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.x_norm = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        else:
            # Use running statistics during inference
            self.x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        self.output = self.gamma * self.x_norm + self.beta
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for batch normalization
        """
        batch_size = self.input.shape[0]
        
        # Gradients for gamma and beta
        dgamma = np.sum(output_gradient * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient for normalized input
        dx_norm = output_gradient * self.gamma
        
        # Gradient for variance
        dvar = np.sum(dx_norm * (self.input - self.batch_mean) * -0.5 * 
                      np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        
        # Gradient for mean
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)
        
        # Gradient for input
        dx = dx_norm / np.sqrt(self.batch_var + self.epsilon) + \
             dvar * 2 * (self.input - self.batch_mean) / batch_size + \
             dmean / batch_size
        
        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx
    
    def get_params(self) -> Dict:
        """
        Get layer parameters for saving
        """
        return {
            'type': 'BatchNormalization',
            'gamma': self.gamma.tolist(),
            'beta': self.beta.tolist(),
            'running_mean': self.running_mean.tolist(),
            'running_var': self.running_var.tolist(),
            'epsilon': self.epsilon,
            'momentum': self.momentum
        }
    
    def set_params(self, params: Dict) -> None:
        """
        Set layer parameters from loaded data
        """
        self.gamma = np.array(params['gamma'])
        self.beta = np.array(params['beta'])
        self.running_mean = np.array(params['running_mean'])
        self.running_var = np.array(params['running_var'])
        self.epsilon = params['epsilon']
        self.momentum = params['momentum']


class NeuralNetwork:
    """
    Neural Network class that combines layers and implements training
    """
    def __init__(self, loss: str = 'mse'):
        self.layers = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        # Set loss function and its derivative
        self.loss_name = loss
        if loss == 'mse':
            self.loss = Loss.mse
            self.loss_derivative = Loss.mse_derivative
        elif loss == 'binary_crossentropy':
            self.loss = Loss.binary_crossentropy
            self.loss_derivative = Loss.binary_crossentropy_derivative
        elif loss == 'categorical_crossentropy':
            self.loss = Loss.categorical_crossentropy
            self.loss_derivative = Loss.categorical_crossentropy_derivative
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
    
    def add(self, layer: Layer) -> None:
        """
        Add a layer to the network
        """
        self.layers.append(layer)
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network for prediction
        """
        # Set all layers to inference mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        # Forward pass
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> float:
        """
        Train the network on a single batch
        """
        # Set all layers to training mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        # Forward pass
        output = x_batch
        for layer in self.layers:
            output = layer.forward(output)
        
        # Calculate loss
        loss_value = self.loss(y_batch, output)
        
        # Add regularization loss if any
        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'reg_fn'):
                reg_loss += layer.reg_fn()
        
        total_loss = loss_value + reg_loss
        
        # Backward pass
        gradient = self.loss_derivative(y_batch, output)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        
        return total_loss
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 1000, batch_size: int = 32, learning_rate: float = 0.01,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            lr_scheduler: Optional[Callable[[float, int], float]] = None,
            early_stopping: Optional[int] = None,
            verbose: bool = True) -> Dict:
        """
        Train the network on the given dataset
        """
        num_samples = x_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Initialize learning rate scheduler if provided
        if lr_scheduler is None:
            lr_scheduler = LearningRateScheduler.constant
        
        # Training loop
        for epoch in range(epochs):
            # Update learning rate based on scheduler
            current_lr = lr_scheduler(learning_rate, epoch)
            
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = self.train_on_batch(x_batch, y_batch, current_lr)
                epoch_loss += batch_loss * len(x_batch)
            
            # Calculate average loss
            epoch_loss /= num_samples
            self.history['loss'].append(epoch_loss)
            
            # Calculate training accuracy if classification task
            if self.loss_name in ['binary_crossentropy', 'categorical_crossentropy']:
                y_pred = self.predict(x_train)
                accuracy = self._calculate_accuracy(y_train, y_pred)
                self.history['accuracy'].append(accuracy)
            
            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                y_val_pred = self.predict(x_val)
                val_loss = self.loss(y_val, y_val_pred)
                self.history['val_loss'].append(val_loss)
                
                # Calculate validation accuracy if classification task
                if self.loss_name in ['binary_crossentropy', 'categorical_crossentropy']:
                    val_accuracy = self._calculate_accuracy(y_val, y_val_pred)
                    self.history['val_accuracy'].append(val_accuracy)
                
                # Early stopping
                if early_stopping is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                status = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}"
                if 'accuracy' in self.history and self.history['accuracy']:
                    status += f", Accuracy: {self.history['accuracy'][-1]:.4f}"
                if 'val_loss' in self.history and self.history['val_loss']:
                    status += f", Val Loss: {self.history['val_loss'][-1]:.4f}"
                if 'val_accuracy' in self.history and self.history['val_accuracy']:
                    status += f", Val Accuracy: {self.history['val_accuracy'][-1]:.4f}"
                print(status)
        
        return self.history
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy for classification tasks
        """
        if self.loss_name == 'binary_crossentropy':
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        elif self.loss_name == 'categorical_crossentropy':
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_classes)
        else:
            # Not a classification task
            return 0.0
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model parameters to a file
        """
        model_params = {
            'loss': self.loss_name,
            'layers': [layer.get_params() for layer in self.layers]
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_params, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model parameters from a file
        """
        with open(filepath, 'r') as f:
            model_params = json.load(f)
        
        # Set loss function
        self.loss_name = model_params['loss']
        if self.loss_name == 'mse':
            self.loss = Loss.mse
            self.loss_derivative = Loss.mse_derivative
        elif self.loss_name == 'binary_crossentropy':
            self.loss = Loss.binary_crossentropy
            self.loss_derivative = Loss.binary_crossentropy_derivative
        elif self.loss_name == 'categorical_crossentropy':
            self.loss = Loss.categorical_crossentropy
            self.loss_derivative = Loss.categorical_crossentropy_derivative
        
        # Create layers
        self.layers = []
        for layer_params in model_params['layers']:
            layer_type = layer_params['type']
            if layer_type == 'Dense':
                weights = np.array(layer_params['weights'])
                input_size, output_size = weights.shape
                layer = Dense(input_size, output_size, activation=layer_params['activation'])
                layer.set_params(layer_params)
                self.layers.append(layer)
            elif layer_type == 'Dropout':
                layer = Dropout(rate=layer_params['rate'])
                self.layers.append(layer)
            elif layer_type == 'BatchNormalization':
                input_shape = len(layer_params['gamma'][0])
                layer = BatchNormalization(input_shape)
                layer.set_params(layer_params)
                self.layers.append(layer)
    
    def plot_history(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot training history
        """
        plt.figure(figsize=figsize)
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy if available
        if 'accuracy' in self.history and self.history['accuracy']:
            plt.subplot(1, 2, 2)
            plt.plot(self.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history and self.history['val_accuracy']:
                plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.show()


class DataGenerator:
    """
    Class for generating synthetic datasets for testing neural networks
    """
    @staticmethod
    def generate_xor_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate XOR problem dataset
        """
        # Generate random binary inputs
        X = np.random.randint(0, 2, size=(n_samples, 2))
        
        # XOR operation: output is 1 if inputs are different, 0 if they are the same
        y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)
        
        return X, y
    
    @staticmethod
    def generate_circle_data(n_samples: int = 1000, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate circular classification dataset
        """
        # Generate random points in a circle
        r = np.random.uniform(0, 1, size=n_samples)
        theta = np.random.uniform(0, 2*np.pi, size=n_samples)
        X = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta)
        ])
        
        # Add noise
        X += np.random.normal(0, noise, X.shape)
        
        # Label: 1 if point is in inner circle, 0 otherwise
        y = (r < 0.5).astype(int).reshape(-1, 1)
        
        return X, y
    
    @staticmethod
    def generate_spiral_data(n_samples: int = 1000, n_classes: int = 2, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spiral classification dataset
        """
        X = np.zeros((n_samples * n_classes, 2))
        y = np.zeros(n_samples * n_classes, dtype=int)
        
        for j in range(n_classes):
            ix = range(n_samples * j, n_samples * (j + 1))
            r = np.linspace(0, 1, n_samples)
            t = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.normal(0, noise, n_samples)
            X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
            y[ix] = j
        
        # One-hot encode the labels if more than 2 classes
        if n_classes > 2:
            y_one_hot = np.zeros((n_samples * n_classes, n_classes))
            for i, label in enumerate(y):
                y_one_hot[i, label] = 1
            return X, y_one_hot
        else:
            return X, y.reshape(-1, 1)
    
    @staticmethod
    def generate_regression_data(n_samples: int = 1000, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate regression dataset
        """
        X = np.random.uniform(-1, 1, size=(n_samples, 1))
        y = np.sin(2 * np.pi * X) + np.random.normal(0, noise, size=(n_samples, 1))
        
        return X, y


class ModelVisualizer:
    """
    Class for visualizing neural network predictions
    """
    @staticmethod
    def plot_decision_boundary(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, 
                              h: float = 0.01, cmap: str = 'viridis') -> None:
        """
        Plot decision boundary for 2D classification problems
        """
        # Set min and max values with some padding
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        
        # Create a mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Predict on mesh grid points
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Process predictions based on output shape
        if Z.shape[1] > 1:  # Multi-class classification
            Z = np.argmax(Z, axis=1)
        else:  # Binary classification
            Z = (Z > 0.5).astype(int).ravel()
        
        # Reshape to match the mesh grid
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
        
        # Plot training points
        if y.shape[1] > 1:  # One-hot encoded labels
            scatter = plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), 
                                 edgecolors='k', cmap=cmap, alpha=1)
        else:  # Binary labels
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                                 edgecolors='k', cmap=cmap, alpha=1)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Decision Boundary')
        plt.colorbar(scatter)
        plt.show()
    
    @staticmethod
    def plot_regression_prediction(model: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot regression predictions
        """
        # Sort X for smooth line plotting
        sort_idx = np.argsort(X.ravel())
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Make predictions
        y_pred = model.predict(X_sorted)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
        plt.plot(X_sorted, y_pred, color='red', linewidth=2, label='Prediction')
        plt.title('Regression Prediction')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()