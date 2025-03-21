import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_cdf_difference(y_points, x):
    mean = 0.8 * x
    std = 1.0
    cdf_values = norm.cdf(y_points, loc=mean, scale=std)
    cdf_diff = np.diff(cdf_values)
    return cdf_diff

def numerical_conditional_expectation(x, g_func, y_min=-10, y_max=10, n_points=1000):
    y_points = np.linspace(y_min, y_max, n_points)
    g_values = g_func(y_points)
    cdf_diffs = compute_cdf_difference(y_points, x)
    g_avg = (g_values[1:] + g_values[:-1]) / 2
    result = np.sum(g_avg * cdf_diffs)
    return result

def g1(y):
    return y

def g2(y):
    return np.clip(y, -1, 1)

def compute_expectations_over_range(x_range):
    e1 = np.array([numerical_conditional_expectation(x, g1) for x in x_range])
    e2 = np.array([numerical_conditional_expectation(x, g2) for x in x_range])
    return e1, e2

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

def generate_data(N=500, x_min=-5, x_max=5):
    X = np.random.uniform(x_min, x_max, N)
    W = np.random.standard_normal(N)
    Y = 0.8 * X + W
    return X, Y

def criterion_A1(pred, target):
    return torch.mean(0.5 * pred**2 - target * pred)

def criterion_A2(pred, target):
    phi = (torch.exp(0.5 * torch.abs(pred)) - 1) + (1/3) * (torch.exp(-1.5 * torch.abs(pred)) - 1)
    psi = 2 * torch.sign(pred) * (torch.exp(-0.5 * torch.abs(pred)) - 1)
    return torch.mean(phi + target * psi)

def criterion_C1(pred, target, a=-1, b=1):
    z = pred
    z = torch.sigmoid(z)
    z = pred
    phi = (b - a)/(1 + torch.exp(z)) + b * torch.log(1 + torch.exp(z))
    psi = -torch.log(1 + torch.exp(z))
    return torch.mean(phi + target * psi)

def train_networks(X_train, Y_train, hidden_size=50, epochs=1000, bounded=True):
    X_train = torch.FloatTensor(X_train.reshape(-1, 1))
    if bounded:
        Y_train = np.clip(Y_train, -1, 1)
    Y_train = torch.FloatTensor(Y_train.reshape(-1, 1))
    
    net_A1 = ShallowNetwork(hidden_size)
    net_A2 = ShallowNetwork(hidden_size) if not bounded else ShallowNetwork(hidden_size)
    
    opt_A1 = optim.Adam(net_A1.parameters())
    opt_A2 = optim.Adam(net_A2.parameters())
    
    losses_A1 = []
    losses_A2 = []
    
    for epoch in range(epochs):
        opt_A1.zero_grad()
        pred_A1 = torch.tanh(net_A1(X_train)) if bounded else net_A1(X_train)
        loss_A1 = criterion_A1(pred_A1, Y_train)
        loss_A1.backward()
        opt_A1.step()
        
        opt_A2.zero_grad()
        pred_A2 = net_A2(X_train)
        loss_A2 = criterion_C1(pred_A2, Y_train, a=-1, b=1) if bounded else criterion_A2(pred_A2, Y_train)
        loss_A2.backward()
        opt_A2.step()
        
        if epoch % 100 == 0:
            losses_A1.append(loss_A1.item())
            losses_A2.append(loss_A2.item())
    
    return net_A1, net_A2, losses_A1, losses_A2

# ===== Plotting Functions =====
def plot_combined_results(X, Y, nets_unbounded, nets_bounded, x_range):
    # Sort X 
    X_sorted_idx = np.argsort(X)
    X_sorted = X[X_sorted_idx]
    
    # Get neural network 
    X_tensor = torch.FloatTensor(X_sorted.reshape(-1, 1))
    with torch.no_grad():
        net_A1_unbounded, net_A2_unbounded = nets_unbounded
        Y_pred_A1_unbounded = net_A1_unbounded(X_tensor).numpy()
        Y_pred_A2_unbounded = net_A2_unbounded(X_tensor).numpy()
        
        net_A1_bounded, net_C1_bounded = nets_bounded
        Y_pred_A1_bounded = torch.tanh(net_A1_bounded(X_tensor)).numpy()
        Y_pred_C1_bounded = torch.sigmoid(net_C1_bounded(X_tensor)).numpy()
        Y_pred_C1_bounded = -1 + 2 * Y_pred_C1_bounded
    
    # Get numerical 
    e1, e2 = compute_expectations_over_range(x_range)
    
    plt.figure(figsize=(15, 5))
    
    # Unbounded 
    plt.subplot(1, 2, 1)
    plt.scatter(X, Y, alpha=0.3, label='Data')
    plt.plot(x_range, e1, 'r-', label='Numerical E[Y|X]')
    plt.plot(X_sorted, Y_pred_A1_unbounded, 'g--', label='NN A1 Prediction')
    plt.plot(X_sorted, Y_pred_A2_unbounded, 'b--', label='NN A2 Prediction')
    plt.title('Unbounded Conditional Expectation')
    plt.legend()
    plt.grid(True)
    
    # Bounded 
    plt.subplot(1, 2, 2)
    plt.scatter(X, np.clip(Y, -1, 1), alpha=0.3, label='Data')
    plt.plot(x_range, e2, 'r-', label='Numerical E[clip(Y)]')
    plt.plot(X_sorted, Y_pred_A1_bounded, 'g--', label='NN A1 Prediction')
    plt.plot(X_sorted, Y_pred_C1_bounded, 'b--', label='NN C1 Prediction')
    plt.title('Bounded Conditional Expectation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_training_losses(losses_A1_unbounded, losses_A2_unbounded, losses_A1_bounded, losses_C1_bounded):
    plt.figure(figsize=(15, 5))
    
    # Unbounded 
    plt.subplot(1, 2, 1)
    plt.plot(losses_A1_unbounded, 'g-', label='NN A1 Loss')
    plt.plot(losses_A2_unbounded, 'b-', label='NN A2 Loss')
    plt.title('Unbounded Training Losses')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Bounded 
    plt.subplot(1, 2, 2)
    plt.plot(losses_A1_bounded, 'g-', label='NN A1 Loss')
    plt.plot(losses_C1_bounded, 'b-', label='NN C1 Loss')
    plt.title('Bounded Training Losses')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X, Y = generate_data(N=500, x_min=-5, x_max=5)
    
    # Train neural networks
    net_A1_unbounded, net_A2_unbounded, losses_A1_unbounded, losses_A2_unbounded = train_networks(X, Y, bounded=False)
    net_A1_bounded, net_C1_bounded, losses_A1_bounded, losses_C1_bounded = train_networks(X, Y, bounded=True)
    
    # Pack networks
    nets_unbounded = (net_A1_unbounded, net_A2_unbounded)
    nets_bounded = (net_A1_bounded, net_C1_bounded)
    
    # Create a custom x range based on sample min/max
    x_lower = int(np.floor(np.min(X)))
    x_upper = int(np.ceil(np.max(X)))
    x_range = np.linspace(x_lower, x_upper, 100)
    
    # Plot combined results
    plot_combined_results(X, Y, nets_unbounded, nets_bounded, x_range)
    
    # Plot training losses
    plot_training_losses(losses_A1_unbounded, losses_A2_unbounded, losses_A1_bounded, losses_C1_bounded)