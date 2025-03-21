import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ShallowNetwork(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def generate_data(N=500):
    # Generate X from standard normal
    X = np.random.standard_normal(N)
    # Generate W from standard normal
    W = np.random.standard_normal(N)
    # Generate Y according to the model Y = 0.8X + W
    Y = 0.8 * X + W
    
    return X, Y

def criterion_A1(pred, target):
    """[A1] criterion: ω(z) = z²/2, ε(z) = -z"""
    return torch.mean(0.5 * pred**2 - target * pred)

def criterion_A2(pred, target):
    """[A2] criterion: ω(z) = (e^(0.5|z|) - 1) + (1/3)(e^(-1.5|z|) - 1),
       ε(z) = 2*sign(z)(e^(-0.5|z|) - 1)"""
    omega = (torch.exp(0.5 * torch.abs(pred)) - 1) + (1/3) * (torch.exp(-1.5 * torch.abs(pred)) - 1)
    epsilon = 2 * torch.sign(pred) * (torch.exp(-0.5 * torch.abs(pred)) - 1)
    return torch.mean(omega + target * epsilon)

def criterion_C1(pred, target, a=-1, b=1):
    """[C1] criterion for bounded interval [a,b]"""
    z = pred  # network output
    sigmoid = torch.sigmoid(z)
    theta = a * (1 - sigmoid) + b * sigmoid
    omega = (b - a)/(1 + torch.exp(z)) + b * torch.log(1 + torch.exp(z))
    epsilon = -torch.log(1 + torch.exp(z))
    return torch.mean(omega + target * epsilon)

def train_unbounded_network(X_train, Y_train, hidden_size=50, epochs=1000):
    X_train = torch.FloatTensor(X_train.reshape(-1, 1))
    Y_train = torch.FloatTensor(Y_train.reshape(-1, 1))
    
    # Create two networks for A1 and A2
    net_A1 = ShallowNetwork(hidden_size)
    net_A2 = ShallowNetwork(hidden_size)
    
    opt_A1 = optim.Adam(net_A1.parameters())
    opt_A2 = optim.Adam(net_A2.parameters())
    
    losses_A1 = []
    losses_A2 = []
    
    for epoch in range(epochs):
        # Train with A1 criterion
        opt_A1.zero_grad()
        pred_A1 = net_A1(X_train)
        loss_A1 = criterion_A1(pred_A1, Y_train)
        loss_A1.backward()
        opt_A1.step()
        
        # Train with A2 criterion
        opt_A2.zero_grad()
        pred_A2 = net_A2(X_train)
        loss_A2 = criterion_A2(pred_A2, Y_train)
        loss_A2.backward()
        opt_A2.step()
        
        if epoch % 100 == 0:
            losses_A1.append(loss_A1.item())
            losses_A2.append(loss_A2.item())
    
    return net_A1, net_A2, losses_A1, losses_A2

def train_bounded_network(X_train, Y_train, hidden_size=50, epochs=1000):
    X_train = torch.FloatTensor(X_train.reshape(-1, 1))
    # Clip Y values to [-1, 1] before training
    Y_train = np.clip(Y_train, -1, 1)
    Y_train = torch.FloatTensor(Y_train.reshape(-1, 1))
    
    # Create two networks for A1 and C1
    net_A1 = ShallowNetwork(hidden_size)
    net_C1 = ShallowNetwork(hidden_size)
    
    opt_A1 = optim.Adam(net_A1.parameters())
    opt_C1 = optim.Adam(net_C1.parameters())
    
    losses_A1 = []
    losses_C1 = []
    
    for epoch in range(epochs):
        # Train with A1 criterion
        opt_A1.zero_grad()
        pred_A1 = torch.tanh(net_A1(X_train))  # Use tanh to bound output
        loss_A1 = criterion_A1(pred_A1, Y_train)
        loss_A1.backward()
        opt_A1.step()
        
        # Train with C1 criterion
        opt_C1.zero_grad()
        pred_C1 = net_C1(X_train)
        loss_C1 = criterion_C1(pred_C1, Y_train, a=-1, b=1)
        loss_C1.backward()
        opt_C1.step()
        
        if epoch % 100 == 0:
            losses_A1.append(loss_A1.item())
            losses_C1.append(loss_C1.item())
    
    return net_A1, net_C1, losses_A1, losses_C1

def plot_results(X, Y, nets_unbounded, nets_bounded):
    # Sort X for smooth plotting
    X_sorted_idx = np.argsort(X)
    X_sorted = X[X_sorted_idx]
    Y_sorted = Y[X_sorted_idx]
    
    X_tensor = torch.FloatTensor(X_sorted.reshape(-1, 1))
    
    # Get predictions
    with torch.no_grad():
        net_A1_unbounded, net_A2_unbounded = nets_unbounded
        Y_pred_A1_unbounded = net_A1_unbounded(X_tensor).numpy()
        Y_pred_A2_unbounded = net_A2_unbounded(X_tensor).numpy()
        
        net_A1_bounded, net_C1_bounded = nets_bounded
        Y_pred_A1_bounded = torch.tanh(net_A1_bounded(X_tensor)).numpy()
        Y_pred_C1_bounded = torch.sigmoid(net_C1_bounded(X_tensor)).numpy()
        Y_pred_C1_bounded = -1 + 2 * Y_pred_C1_bounded  # Scale to [-1, 1]
    
    # True conditional expectations
    Y_true_unbounded = 0.8 * X_sorted
    Y_true_bounded = np.clip(0.8 * X_sorted, -1, 1)
    
    plt.figure(figsize=(15, 5))
    
    # Unbounded case
    plt.subplot(1, 2, 1)
    plt.scatter(X, Y, alpha=0.3, label='Data')
    plt.plot(X_sorted, Y_true_unbounded, 'r-', label='True E[Y|X]')
    plt.plot(X_sorted, Y_pred_A1_unbounded, 'g--', label='A1 Prediction')
    plt.plot(X_sorted, Y_pred_A2_unbounded, 'b--', label='A2 Prediction')
    plt.title('Unbounded Conditional Expectation')
    plt.legend()
    plt.grid(True)
    
    # Bounded case
    plt.subplot(1, 2, 2)
    plt.scatter(X, np.clip(Y, -1, 1), alpha=0.3, label='Data')
    plt.plot(X_sorted, Y_true_bounded, 'r-', label='True E[min{1,max{-1,Y}}|X]')
    plt.plot(X_sorted, Y_pred_A1_bounded, 'g--', label='A1 Prediction')
    plt.plot(X_sorted, Y_pred_C1_bounded, 'b--', label='C1 Prediction')
    plt.title('Bounded Conditional Expectation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(losses_unbounded, losses_bounded):
    losses_A1_unbounded, losses_A2_unbounded = losses_unbounded
    losses_A1_bounded, losses_C1_bounded = losses_bounded
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_A1_unbounded, label='A1')
    plt.plot(losses_A2_unbounded, label='A2')
    plt.title('Unbounded Network Learning Curves')
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_A1_bounded, label='A1')
    plt.plot(losses_C1_bounded, label='C1')
    plt.title('Bounded Network Learning Curves')
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X, Y = generate_data(N=500)
    
    # Train networks - now correctly unpacking 4 values
    net_A1_unbounded, net_A2_unbounded, losses_A1_unbounded, losses_A2_unbounded = train_unbounded_network(X, Y, hidden_size=50)
    net_A1_bounded, net_C1_bounded, losses_A1_bounded, losses_C1_bounded = train_bounded_network(X, Y, hidden_size=50)
    
    # Pack networks and losses for plotting functions
    nets_unbounded = (net_A1_unbounded, net_A2_unbounded)
    nets_bounded = (net_A1_bounded, net_C1_bounded)
    losses_unbounded = (losses_A1_unbounded, losses_A2_unbounded)
    losses_bounded = (losses_A1_bounded, losses_C1_bounded)
    
    # Plot results
    plot_results(X, Y, nets_unbounded, nets_bounded)
    
    # Plot learning curves
    plot_learning_curves(losses_unbounded, losses_bounded)