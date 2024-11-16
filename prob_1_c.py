import numpy as np

# Hyperparameters
input_size = 2        # x1 and x2
hidden_size = 20      # Hidden layer size
output_size = 1       # Output layer size
learning_rate = 0.001  # Learning rate
epochs = 1000         # Number of training epochs

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generating the training data for the condition cosh(x1) * cosh(x2) > e
def generate_data(num_samples=1000):
    X = np.random.uniform(-3, 3, (num_samples, 2))  # Random x1, x2 values between -3 and 3
    y = np.zeros(num_samples)
    for i in range(num_samples):
        x1, x2 = X[i]
        if np.cosh(x1) * np.cosh(x2) > np.e:
            y[i] = 1
    return X, y

# Neural Network initialization
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for input to hidden
    b1 = np.zeros((1, hidden_size))  # Biases for hidden layer
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for hidden to output
    b2 = np.zeros((1, output_size))  # Biases for output layer
    return W1, b1, W2, b2

# Forward pass
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)  # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # Output layer activation
    return A1, A2

# Training the neural network using cross-entropy method
def train_nn(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs):
    # Initialize weights
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        A1, A2 = forward(X_train, W1, b1, W2, b2)

        # Calculate loss
        loss = cross_entropy_loss(y_train, A2)
        
        # Backpropagation
        dA2 = A2 - y_train.reshape(-1, 1)  # Reshaping y_train to match output shape
        dW2 = np.dot(A1.T, dA2)  # Gradient for weights between hidden and output layer
        db2 = np.sum(dA2, axis=0, keepdims=True)  # Gradient for bias of output layer

        dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)  # Gradient for hidden layer
        dW1 = np.dot(X_train.T, dA1)  # Gradient for weights between input and hidden layer
        db1 = np.sum(dA1, axis=0, keepdims=True)  # Gradient for bias of hidden layer

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

# Generate training data
X_train, y_train = generate_data()

# Train the neural network
W1, b1, W2, b2 = train_nn(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs)

# Testing the neural network
def test_nn(X_test, W1, b1, W2, b2):
    _, A2 = forward(X_test, W1, b1, W2, b2)
    return A2

# Example test
X_test = np.array([[1.0, 2.0], [0.5, -0.5], [-1.5, 2.0], [-2.0, -2.0]])
predictions = test_nn(X_test, W1, b1, W2, b2)
print("Predictions (probabilities):", predictions)
