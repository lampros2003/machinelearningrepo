import numpy as np

# Hyperparameters
input_size = 2        # x1 and x2
hidden_size = 20      # Hidden layer size
output_size = 1       # Output layer size
learning_rate = 0.0001  # Learning rate
epochs = 3000       # Number of training epochs
input_size = 2        # x1 and x2
hidden_size = 20      # Hidden layer size
output_size = 1       # Output layer size
  # Learning rate
beta1 = 0.5          # Exponential decay rate for the first moment estimate
beta2 = 0.9        # Exponential decay rate for the second moment estimate
epsilon = 1e-8        # Small value to prevent division by zero

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
def generate_data(num_samples=200):
    newx0_1, newx0_2 = np.random.normal(0, 1, (2, num_samples))
    # Generate samples from H1

    newx1_1 =  0.5*(np.random.normal(1, 1, num_samples)+np.random.normal(-1, 1, num_samples))
    newx1_2 =  0.5*(np.random.normal(1, 1, num_samples)+np.random.normal(-1, 1, num_samples))
    H_0 = np.append(newx0_1, newx0_2) 
    H_1 = np.append(newx1_1, newx1_2)
    X = np.column_stack((H_0, H_1))
    
    y = np.zeros(num_samples)
    y = np.append(y, np.ones(num_samples))
    indexes = list(range(2*num_samples))
    np.random.shuffle(indexes)
    for i in indexes:
        x1, x2 = X[i]
        if np.cosh(x1) * np.cosh(x2) > np.e:
            y[i] = 1
    return X, y

# Neural Network initialization
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 =((1/(input_size*hidden_size))) *np.random.randn(input_size, hidden_size)  # Weights for input to hidden
    b1 = np.zeros((1, hidden_size))  # Biases for hidden layer
    W2 = ((1/(input_size*hidden_size))) *np.random.randn(hidden_size, output_size)   # Weights for hidden to output
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
def train_nn(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs, beta1, beta2, epsilon):
    # Initialize weights
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    
    # Initialize moment estimates
    mW1, mb1, mW2, mb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
    vW1, vb1, vW2, vb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
    
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

        # Update the moment estimates
        mW1 = beta1 * mW1 + (1 - beta1) * dW1
        mb1 = beta1 * mb1 + (1 - beta1) * db1
        mW2 = beta1 * mW2 + (1 - beta1) * dW2
        mb2 = beta1 * mb2 + (1 - beta1) * db2
        
        vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
        vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
        vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
        vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)

        # Bias correction
        mW1_hat = mW1 / (1 - beta1**(epoch + 1))
        mb1_hat = mb1 / (1 - beta1**(epoch + 1))
        mW2_hat = mW2 / (1 - beta1**(epoch + 1))
        mb2_hat = mb2 / (1 - beta1**(epoch + 1))
        
        vW1_hat = vW1 / (1 - beta2**(epoch + 1))
        vb1_hat = vb1 / (1 - beta2**(epoch + 1))
        vW2_hat = vW2 / (1 - beta2**(epoch + 1))
        vb2_hat = vb2 / (1 - beta2**(epoch + 1))

        # Update weights
        W1 -= learning_rate * mW1_hat / (np.sqrt(vW1_hat) + epsilon)
        b1 -= learning_rate * mb1_hat / (np.sqrt(vb1_hat) + epsilon)
        W2 -= learning_rate * mW2_hat / (np.sqrt(vW2_hat) + epsilon)
        b2 -= learning_rate * mb2_hat / (np.sqrt(vb2_hat) + epsilon)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

# Generate training data
X_train, y_train = generate_data()


# Train the neural network
W1, b1, W2, b2 = train_nn(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs, beta1, beta2, epsilon)

# Testing the neural network
def test_nn(X_test, W1, b1, W2, b2):
    _, A2 = forward(X_test, W1, b1, W2, b2)
    return A2

def calculate_error_rate(X, y_true, W1, b1, W2, b2, threshold=0.5):
    """
    Calculate the error rate of the neural network on a given dataset.
    
    Parameters:
    - X: Input data (num_samples, num_features).
    - y_true: True labels (num_samples, 1).
    - W1, b1, W2, b2: Trained weights and biases of the neural network.
    - threshold: Probability threshold for converting probabilities to binary predictions (default is 0.5).
    
    Returns:
    - error_rate: The error rate of the model on the given dataset.
    """
    # Ensure that the shapes of X and y_true are compatible
    assert X.shape[0] == y_true.shape[0], "Number of samples in X and y_true must match"
    
    # Get the model predictions (probabilities)
    predictions = test_nn(X, W1, b1, W2, b2)
    
    # Flatten predictions to avoid shape issues
    predictions = predictions.flatten()
    
    # Convert probabilities to binary predictions using the threshold
    binary_predictions = (predictions > threshold).astype(int)
    
    # Calculate the number of incorrect predictions
    incorrect_predictions = np.sum(binary_predictions != y_true)
    
    # Calculate the error rate
    error_rate = incorrect_predictions / len(y_true)
    
    return error_rate
# Example test

# Generate test data`\


X_0_test = np.append(x0_1, x0_2)
X_1_test = np.append(x1_1, x1_2)
X_test = np.column_stack((X_0_test, X_1_test)) 
y0_test = np.zeros(10**6)
y1_test = np.ones(10**6)
y_test = np.append(y0_test, y1_test)
indexes = list(range(2*10**6))  
np.random.shuffle(indexes)

X_test = X_test[indexes]
y_test = y_test[indexes]

er = calculate_error_rate(X_test, y_test, W1, b1, W2, b2)

print("Error rate:", er)
