import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_data(N):
    """Generate training data according to the model Y = 0.8X + W"""
    X = np.random.standard_normal(N)
    W = np.random.standard_normal(N)
    Y = 0.8 * X + W
    return X, Y

# Neural Network architecture
class ConditionalExpectationNet(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x.reshape(-1, 1)).squeeze()

# Training function using [A1] criterion (MSE)
def train_network(X, Y, model, target_function, epochs=1000, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X_tensor)
        
        # Compute target values
        target = torch.FloatTensor([target_function(y) for y in Y_tensor])
        
        # MSE loss ([A1] criterion)
        loss = torch.mean((pred - target) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            losses.append(loss.item())
    
    return losses

# Numerical solution functions
def compute_numerical_expectation(x_point, n_samples=10000):
    """Compute E[Y|X = x] numerically"""
    w_samples = np.random.standard_normal(n_samples)
    y_samples = 0.8 * x_point + w_samples
    return np.mean(y_samples)

def compute_numerical_clipped_expectation(x_point, n_samples=10000):
    """Compute E[min{1, max{-1, Y}}|X = x] numerically"""
    w_samples = np.random.standard_normal(n_samples)
    y_samples = 0.8 * x_point + w_samples
    clipped_samples = np.clip(y_samples, -1, 1)
    return np.mean(clipped_samples)

# Generate training data
N = 500
X_train, Y_train = generate_data(N)

# Train models for both conditional expectations
model1 = ConditionalExpectationNet(hidden_size=50)
model2 = ConditionalExpectationNet(hidden_size=50)

# Train for E[Y|X]
losses1 = train_network(X_train, Y_train, model1, lambda y: y)

# Train for E[min{1, max{-1, Y}}|X]
losses2 = train_network(X_train, Y_train, model2, lambda y: max(-1, min(1, y)))

# Compute and plot results
x_test = np.linspace(min(X_train), max(X_train), 100)
numerical_expectations = [compute_numerical_expectation(x) for x in x_test]
numerical_clipped = [compute_numerical_clipped_expectation(x) for x in x_test]

model1.eval()
model2.eval()
with torch.no_grad():
    nn_expectations = model1(torch.FloatTensor(x_test)).numpy()
    nn_clipped = model2(torch.FloatTensor(x_test)).numpy()

plt.figure(figsize=(12, 5))

# Plot first conditional expectation
plt.subplot(1, 2, 1)
plt.plot(x_test, numerical_expectations, 'b-', label='Numerical')
plt.plot(x_test, nn_expectations, 'r--', label='Neural Network')
plt.title('E[Y|X]')
plt.xlabel('X')
plt.ylabel('Conditional Expectation')
plt.legend()
plt.grid(True)

# Plot second conditional expectation
plt.subplot(1, 2, 2)
plt.plot(x_test, numerical_clipped, 'b-', label='Numerical')
plt.plot(x_test, nn_clipped, 'r--', label='Neural Network')
plt.title('E[min{1, max{-1, Y}}|X]')
plt.xlabel('X')
plt.ylabel('Conditional Expectation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses1)
plt.title('Learning Curve - E[Y|X]')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses2)
plt.title('Learning Curve - E[min{1, max{-1, Y}}|X]')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()