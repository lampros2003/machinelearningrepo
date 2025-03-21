import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import integrate

class ShallowNetwork(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def reward_function(s):
    """R(S) = min{2, S²}"""
    return np.minimum(2, s**2)

def transition_density(s_next, s, action):
    """p(s_next|s,α) - Gaussian transition density"""
    if action == 1:
        mean = 0.8 * s + 1.0
    else:  # action == 2
        mean = -2.0
    
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * (s_next - mean)**2)

def generate_training_data(N=1000):
    """Generate training data as specified in the problem"""
    # Initialize arrays
    S = np.zeros(N+1)  # States
    S_next = np.zeros(N)  # Next states
    actions = np.random.choice([1, 2], size=N)  # Random actions
    
    # Set initial state
    S[0] = np.random.standard_normal()
    
    # Generate trajectories
    for t in range(N):
        if actions[t] == 1:
            S_next[t] = 0.8 * S[t] + 1.0 + np.random.standard_normal()
        else:
            S_next[t] = -2.0 + np.random.standard_normal()
        
        if t < N:
            S[t+1] = S_next[t]
    
    # Split into action-specific datasets
    action1_mask = actions == 1
    action2_mask = actions == 2
    
    data_action1 = {
        'states': S[:-1][action1_mask],
        'next_states': S_next[action1_mask]
    }
    
    data_action2 = {
        'states': S[:-1][action2_mask],
        'next_states': S_next[action2_mask]
    }
    
    return data_action1, data_action2

# Modify numerical_solution function to use wider state range
def numerical_solution(grid_points=5000):
    """Compute v₁(s) and v₂(s) using trapezoidal integration"""
    s_grid = np.linspace(-20, 20, grid_points)  # Modified range
    ds = s_grid[1] - s_grid[0]
    
    v1 = np.zeros(grid_points)
    v2 = np.zeros(grid_points)
    
    for i, s in enumerate(s_grid):
        integrand1 = reward_function(s_grid) * transition_density(s_grid, s, 1)
        v1[i] = np.trapz(integrand1, s_grid)
        
        integrand2 = reward_function(s_grid) * transition_density(s_grid, s, 2)
        v2[i] = np.trapz(integrand2, s_grid)
    
    return s_grid, v1, v2
def train_networks(data_action1, data_action2, hidden_size=100, epochs=2000):
    """Train neural networks using [A1] and [C1] criteria"""
    # Convert data to PyTorch tensors
    states1 = torch.FloatTensor(data_action1['states'].reshape(-1, 1))
    states2 = torch.FloatTensor(data_action2['states'].reshape(-1, 1))
    
    rewards1 = torch.FloatTensor(reward_function(data_action1['next_states']).reshape(-1, 1))
    rewards2 = torch.FloatTensor(reward_function(data_action2['next_states']).reshape(-1, 1))
    
    # Create networks
    net1 = ShallowNetwork(hidden_size)
    net2 = ShallowNetwork(hidden_size)
    
    # Optimizers
    optimizer1 = optim.Adam(net1.parameters())
    optimizer2 = optim.Adam(net2.parameters())
    
    # Using [C1] since rewards are bounded in [0,2]
    def criterion(pred, target):
        z = pred
        sigmoid = torch.sigmoid(z)
        theta = 2 * sigmoid  # Scale to [0,2]
        omega = 2/(1 + torch.exp(z)) + 2 * torch.log(1 + torch.exp(z))
        epsilon = -torch.log(1 + torch.exp(z))
        return torch.mean(omega + target * epsilon)
    
    losses1 = []
    losses2 = []
    
    for epoch in range(epochs):
        # Train network for action 1
        optimizer1.zero_grad()
        pred1 = net1(states1)
        loss1 = criterion(pred1, rewards1)
        loss1.backward()
        optimizer1.step()
        
        # Train network for action 2
        optimizer2.zero_grad()
        pred2 = net2(states2)
        loss2 = criterion(pred2, rewards2)
        loss2.backward()
        optimizer2.step()
        
        if epoch % 100 == 0:
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            
            # Print progress
            print(f'Epoch {epoch}/{epochs}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')
    
    return net1, net2, losses1, losses2

def plot_value_and_policy(s_grid, v1, v2, net1, net2):
    """Plot value functions and optimal policy"""
    s_tensor = torch.FloatTensor(s_grid.reshape(-1, 1))
    
    with torch.no_grad():
        v1_nn = 2 * torch.sigmoid(net1(s_tensor)).numpy()
        v2_nn = 2 * torch.sigmoid(net2(s_tensor)).numpy()
    
    plt.figure(figsize=(15, 6))
    
    # Plot value functions
    plt.subplot(1, 2, 1)
    plt.plot(s_grid, v1, 'r-', linewidth=2, label='v₁(s) Numerical')
    plt.plot(s_grid, v1_nn, 'r--', linewidth=2, label='v₁(s) Neural Net')
    plt.plot(s_grid, v2, 'b-', linewidth=2, label='v₂(s) Numerical')
    plt.plot(s_grid, v2_nn, 'b--', linewidth=2, label='v₂(s) Neural Net')
    plt.title('Value Functions', fontsize=14)
    plt.xlabel('State (s)', fontsize=12)
    plt.ylabel('Expected Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Plot optimal policy
    plt.subplot(1, 2, 2)
    optimal_action_num = np.where(v1 >= v2, 1, 2)
    optimal_action_nn = np.where(v1_nn >= v2_nn, 1, 2)
    
    plt.plot(s_grid, optimal_action_num, 'r-', linewidth=2, label='Numerical')
    plt.plot(s_grid, optimal_action_nn, 'b--', linewidth=2, label='Neural Net')
    plt.title('Optimal Action Policy', fontsize=14)
    plt.xlabel('State (s)', fontsize=12)
    plt.ylabel('Action', fontsize=12)
    plt.yticks([1, 2])
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_losses(losses1, losses2):
    """Plot training losses separately"""
    plt.figure(figsize=(10, 6))
    epochs = range(0, len(losses1) * 100, 100)
    
    plt.plot(epochs, losses1, 'r-', linewidth=2, label='Action 1 Loss')
    plt.plot(epochs, losses2, 'b-', linewidth=2, label='Action 2 Loss')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    data_action1, data_action2 = generate_training_data(N=1000)
    s_grid, v1, v2 = numerical_solution()
    net1, net2, losses1, losses2 = train_networks(data_action1, data_action2)
    
    plot_value_and_policy(s_grid, v1, v2, net1, net2)
    plot_losses(losses1, losses2)

if __name__ == "__main__":
    main()