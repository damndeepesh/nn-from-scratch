import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import time
from neural_network import NeuralNetwork, Dense, Dropout, BatchNormalization, DataGenerator, ModelVisualizer, LearningRateScheduler

# Set random seed for reproducibility
np.random.seed(42)

# Global variables to store model and data
model = None
X_train, y_train = None, None
X_val, y_val = None, None
training_history = None

# Function to create a neural network model
def create_model(loss, hidden_layers, neurons_per_layer, activation, dropout_rate, batch_norm, regularization_type, regularization_strength):
    global model
    
    # Create a new neural network with the specified loss function
    model = NeuralNetwork(loss=loss)
    
    # Get input and output dimensions based on the dataset
    if X_train is not None:
        input_dim = X_train.shape[1]
        if loss == 'categorical_crossentropy':
            output_dim = y_train.shape[1]
        else:
            output_dim = y_train.shape[1]
    else:
        # Default dimensions if no dataset is loaded
        input_dim = 2
        output_dim = 1
    
    # Parse neurons per layer
    try:
        neurons = [int(n.strip()) for n in neurons_per_layer.split(',')]
        if len(neurons) != hidden_layers:
            neurons = [neurons[0]] * hidden_layers if neurons else [10] * hidden_layers
    except:
        neurons = [10] * hidden_layers
    
    # Set up regularization
    reg = None
    if regularization_type != 'none' and regularization_strength > 0:
        reg = (regularization_type, regularization_strength)
    
    # Add input layer
    if hidden_layers > 0:
        model.add(Dense(input_dim, neurons[0], activation=activation, regularization=reg))
        if batch_norm:
            model.add(BatchNormalization(neurons[0]))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    else:
        # If no hidden layers, connect input directly to output
        model.add(Dense(input_dim, output_dim, activation='sigmoid' if loss == 'binary_crossentropy' else 
                                              'softmax' if loss == 'categorical_crossentropy' else 'linear'))
        return f"Created model with no hidden layers: {input_dim} inputs → {output_dim} outputs"
    
    # Add hidden layers
    for i in range(1, hidden_layers):
        model.add(Dense(neurons[i-1], neurons[i], activation=activation, regularization=reg))
        if batch_norm:
            model.add(BatchNormalization(neurons[i]))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Add output layer with appropriate activation function
    if loss == 'binary_crossentropy':
        output_activation = 'sigmoid'
    elif loss == 'categorical_crossentropy':
        output_activation = 'softmax'
    else:  # MSE
        output_activation = 'linear'
    
    model.add(Dense(neurons[-1], output_dim, activation=output_activation))
    
    # Create a summary of the model architecture
    architecture = f"Created model with {hidden_layers} hidden layers:\n"
    architecture += f"Input({input_dim}) → "
    for i in range(hidden_layers):
        architecture += f"Dense({neurons[i]}, {activation})"
        if batch_norm:
            architecture += " → BatchNorm"
        if dropout_rate > 0:
            architecture += f" → Dropout({dropout_rate})"
        architecture += " → "
    architecture += f"Dense({output_dim}, {output_activation}) → Output"
    
    return architecture

# Function to generate synthetic dataset
def generate_dataset(dataset_type, n_samples, noise, test_split):
    global X_train, y_train, X_val, y_val
    
    # Generate the dataset
    if dataset_type == 'xor':
        X, y = DataGenerator.generate_xor_data(n_samples=n_samples)
    elif dataset_type == 'circle':
        X, y = DataGenerator.generate_circle_data(n_samples=n_samples, noise=noise)
    elif dataset_type == 'spiral':
        X, y = DataGenerator.generate_spiral_data(n_samples=n_samples, noise=noise)
    elif dataset_type == 'spiral_3class':
        X, y = DataGenerator.generate_spiral_data(n_samples=n_samples, n_classes=3, noise=noise)
    elif dataset_type == 'regression':
        X, y = DataGenerator.generate_regression_data(n_samples=n_samples, noise=noise)
    else:
        return "Invalid dataset type"
    
    # Split into train and validation sets
    split_idx = int(n_samples * (1 - test_split))
    indices = np.random.permutation(X.shape[0])
    
    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_val = X[indices[split_idx:]]
    y_val = y[indices[split_idx:]]
    
    # Create a visualization of the dataset
    plt.figure(figsize=(10, 6))
    
    if dataset_type == 'regression':
        plt.scatter(X, y, alpha=0.6)
        plt.title('Regression Dataset')
        plt.xlabel('X')
        plt.ylabel('y')
    else:
        if y.shape[1] > 1:  # One-hot encoded (multiclass)
            plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap='viridis', alpha=0.6)
        else:  # Binary classification
            plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', alpha=0.6)
        plt.title(f'{dataset_type.capitalize()} Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
    
    plt.tight_layout()
    
    # Return information about the dataset
    return (f"Generated {dataset_type} dataset with {n_samples} samples\n"
            f"Training set: {X_train.shape[0]} samples\n"
            f"Validation set: {X_val.shape[0]} samples\n"
            f"Input shape: {X.shape[1]} features\n"
            f"Output shape: {y.shape[1]} {'classes' if dataset_type != 'regression' else 'values'}", 
            plt.gcf())

# Function to train the model
def train_model(epochs, batch_size, learning_rate, lr_scheduler_type, early_stopping):
    global model, X_train, y_train, X_val, y_val, training_history
    
    if model is None:
        return "Please create a model first", None
    
    if X_train is None or y_train is None:
        return "Please generate a dataset first", None
    
    # Set up learning rate scheduler
    if lr_scheduler_type == 'constant':
        lr_scheduler = LearningRateScheduler.constant
    elif lr_scheduler_type == 'step_decay':
        lr_scheduler = lambda lr, epoch: LearningRateScheduler.step_decay(lr, epoch, drop_rate=0.5, epochs_drop=10)
    elif lr_scheduler_type == 'exponential_decay':
        lr_scheduler = lambda lr, epoch: LearningRateScheduler.exponential_decay(lr, epoch, decay_rate=0.95)
    elif lr_scheduler_type == 'time_based_decay':
        lr_scheduler = lambda lr, epoch: LearningRateScheduler.time_based_decay(lr, epoch, decay_rate=0.01)
    else:
        lr_scheduler = None
    
    # Train the model
    start_time = time.time()
    training_history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_data=(X_val, y_val),
        lr_scheduler=lr_scheduler,
        early_stopping=early_stopping if early_stopping > 0 else None,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(training_history['loss'], label='Training Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy if available
    if 'accuracy' in training_history and training_history['accuracy']:
        plt.subplot(1, 2, 2)
        plt.plot(training_history['accuracy'], label='Training Accuracy')
        plt.plot(training_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    
    # Calculate final metrics
    final_train_loss = training_history['loss'][-1]
    final_val_loss = training_history['val_loss'][-1]
    
    metrics_text = f"Training completed in {training_time:.2f} seconds\n"
    metrics_text += f"Final training loss: {final_train_loss:.4f}\n"
    metrics_text += f"Final validation loss: {final_val_loss:.4f}\n"
    
    if 'accuracy' in training_history and training_history['accuracy']:
        final_train_acc = training_history['accuracy'][-1]
        final_val_acc = training_history['val_accuracy'][-1]
        metrics_text += f"Final training accuracy: {final_train_acc:.4f}\n"
        metrics_text += f"Final validation accuracy: {final_val_acc:.4f}"
    
    return metrics_text, plt.gcf()

# Function to visualize model predictions
def visualize_predictions():
    global model, X_train, y_train, X_val, y_val
    
    if model is None:
        return "Please create and train a model first", None
    
    if X_train is None or y_train is None:
        return "Please generate a dataset first", None
    
    # Combine train and validation data for visualization
    X = np.vstack([X_train, X_val])
    y = np.vstack([y_train, y_val])
    
    # Check if it's a regression or classification problem
    if model.loss_name == 'mse':
        # Regression visualization
        ModelVisualizer.plot_regression_prediction(model, X, y)
    else:
        # Classification visualization
        if X.shape[1] == 2:  # Only for 2D inputs
            ModelVisualizer.plot_decision_boundary(model, X, y)
        else:
            return "Decision boundary visualization only works for 2D inputs", None
    
    return "Model predictions visualized", plt.gcf()

# Function to evaluate model on test data
def evaluate_model():
    global model, X_val, y_val
    
    if model is None:
        return "Please create and train a model first"
    
    if X_val is None or y_val is None:
        return "Please generate a dataset first"
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    if model.loss_name == 'mse':
        # Regression metrics
        mse = np.mean(np.square(y_val - y_pred))
        mae = np.mean(np.abs(y_val - y_pred))
        r2 = 1 - (np.sum(np.square(y_val - y_pred)) / np.sum(np.square(y_val - np.mean(y_val))))
        
        return f"Regression Metrics on Validation Set:\n" \
               f"Mean Squared Error: {mse:.4f}\n" \
               f"Mean Absolute Error: {mae:.4f}\n" \
               f"R² Score: {r2:.4f}"
    else:
        # Classification metrics
        if model.loss_name == 'binary_crossentropy':
            # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int)
            accuracy = np.mean(y_pred_class == y_val)
            
            # Calculate precision, recall, and F1 score
            true_positives = np.sum((y_val == 1) & (y_pred_class == 1))
            false_positives = np.sum((y_val == 0) & (y_pred_class == 1))
            false_negatives = np.sum((y_val == 1) & (y_pred_class == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return f"Binary Classification Metrics on Validation Set:\n" \
                   f"Accuracy: {accuracy:.4f}\n" \
                   f"Precision: {precision:.4f}\n" \
                   f"Recall: {recall:.4f}\n" \
                   f"F1 Score: {f1:.4f}"
        else:
            # Multi-class classification
            y_pred_class = np.argmax(y_pred, axis=1)
            y_true_class = np.argmax(y_val, axis=1)
            accuracy = np.mean(y_pred_class == y_true_class)
            
            return f"Multi-class Classification Metrics on Validation Set:\n" \
                   f"Accuracy: {accuracy:.4f}"

# Create the Gradio interface
with gr.Blocks(title="Neural Network from Scratch") as app:
    gr.Markdown("# Neural Network from Scratch")
    gr.Markdown("This application allows you to build, train, and visualize neural networks from scratch without using any deep learning frameworks.")
    
    with gr.Tabs():
        with gr.TabItem("1. Generate Dataset"):
            gr.Markdown("### Step 1: Generate a Synthetic Dataset")
            with gr.Row():
                with gr.Column():
                    dataset_type = gr.Dropdown(
                        choices=["xor", "circle", "spiral", "spiral_3class", "regression"],
                        value="xor",
                        label="Dataset Type"
                    )
                    n_samples = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="Number of Samples")
                    noise = gr.Slider(minimum=0, maximum=0.5, value=0.1, step=0.01, label="Noise Level")
                    test_split = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="Test Split Ratio")
                    generate_btn = gr.Button("Generate Dataset")
                with gr.Column():
                    dataset_info = gr.Textbox(label="Dataset Information", lines=5)
                    dataset_plot = gr.Plot(label="Dataset Visualization")
            
            generate_btn.click(generate_dataset, inputs=[dataset_type, n_samples, noise, test_split], outputs=[dataset_info, dataset_plot])
        
        with gr.TabItem("2. Create Model"):
            gr.Markdown("### Step 2: Create a Neural Network Model")
            with gr.Row():
                with gr.Column():
                    loss_function = gr.Dropdown(
                        choices=["mse", "binary_crossentropy", "categorical_crossentropy"],
                        value="binary_crossentropy",
                        label="Loss Function"
                    )
                    hidden_layers = gr.Slider(minimum=0, maximum=5, value=1, step=1, label="Number of Hidden Layers")
                    neurons_per_layer = gr.Textbox(value="4", label="Neurons Per Layer (comma-separated)")
                    activation = gr.Dropdown(
                        choices=["sigmoid", "tanh", "relu", "leaky_relu"],
                        value="relu",
                        label="Activation Function"
                    )
                    dropout_rate = gr.Slider(minimum=0, maximum=0.5, value=0, step=0.05, label="Dropout Rate")
                    batch_norm = gr.Checkbox(value=False, label="Use Batch Normalization")
                    regularization_type = gr.Dropdown(
                        choices=["none", "l1", "l2"],
                        value="none",
                        label="Regularization Type"
                    )
                    regularization_strength = gr.Slider(minimum=0, maximum=0.1, value=0.01, step=0.001, label="Regularization Strength")
                    create_model_btn = gr.Button("Create Model")
                with gr.Column():
                    model_info = gr.Textbox(label="Model Architecture", lines=5)
            
            create_model_btn.click(
                create_model, 
                inputs=[loss_function, hidden_layers, neurons_per_layer, activation, dropout_rate, 
                        batch_norm, regularization_type, regularization_strength], 
                outputs=[model_info]
            )
        
        with gr.TabItem("3. Train Model"):
            gr.Markdown("### Step 3: Train the Neural Network")
            with gr.Row():
                with gr.Column():
                    epochs = gr.Slider(minimum=10, maximum=2000, value=500, step=10, label="Number of Epochs")
                    batch_size = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Batch Size")
                    learning_rate = gr.Slider(minimum=0.0001, maximum=0.1, value=0.01, step=0.0001, label="Learning Rate")
                    lr_scheduler_type = gr.Dropdown(
                        choices=["constant", "step_decay", "exponential_decay", "time_based_decay"],
                        value="constant",
                        label="Learning Rate Scheduler"
                    )
                    early_stopping = gr.Slider(minimum=0, maximum=100, value=0, step=5, label="Early Stopping Patience (0 to disable)")
                    train_btn = gr.Button("Train Model")
                with gr.Column():
                    training_info = gr.Textbox(label="Training Results", lines=5)
                    training_plot = gr.Plot(label="Training History")
            
            train_btn.click(
                train_model, 
                inputs=[epochs, batch_size, learning_rate, lr_scheduler_type, early_stopping], 
                outputs=[training_info, training_plot]
            )
        
        with gr.TabItem("4. Evaluate & Visualize"):
            gr.Markdown("### Step 4: Evaluate and Visualize the Model")
            with gr.Row():
                with gr.Column(scale=1):
                    evaluate_btn = gr.Button("Evaluate Model")
                    visualize_btn = gr.Button("Visualize Predictions")
                with gr.Column(scale=2):
                    evaluation_info = gr.Textbox(label="Evaluation Metrics", lines=5)
                    visualization_plot = gr.Plot(label="Prediction Visualization")
            
            evaluate_btn.click(evaluate_model, inputs=[], outputs=[evaluation_info])
            visualize_btn.click(visualize_predictions, inputs=[], outputs=[evaluation_info, visualization_plot])

# Launch the app
if __name__ == "__main__":
    app.launch()