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
    S = np.zeros(N+1)
    S_next = np.zeros(N)
    acts = np.random.choice([1, 2], size=N)
   
    # Initialize with wider distribution
    S[0] = np.random.normal(0, 1)
   
    for t in range(N):
        
        if acts[t] == 1:
            S_next[t] = 0.8 * S[t] + 1.0 + np.random.standard_normal()
        else:
            S_next[t] = -2.0 + np.random.standard_normal()
       
        if t < N:
            S[t+1] = S_next[t]
   
    mask1 = acts == 1
    mask2 = acts == 2
   
    data_a1 = {
        'states': S[:-1][mask1],
        'next_states': S_next[mask1]
    }
   
    data_a2 = {
        'states': S[:-1][mask2],
        'next_states': S_next[mask2]
    }
   
    return data_a1, data_a2

# apply the numerical method gicen in the presentationn

def numerical_solution(grid_points=1000, gamma=0.8, max_iter=1000, tol=1e-6):
    """
    Improved numerical solution with wider integration range and better convergence checks
    """
    K = 2
    s_grid = np.linspace(-8, 8, grid_points)
    ds = s_grid[1] - s_grid[0]
   
    V = np.zeros((grid_points, K))
   
    
    R = np.zeros((grid_points, K))
    for j in range(K):
        R[:, j] = reward_function(s_grid)
   
    s_next = np.linspace(-20, 20, grid_points * 2)
    ds_integration = s_next[1] - s_next[0]
   
    for iteration in range(max_iter):
        V_old = V.copy()
       
        F = np.zeros((grid_points, K))
        for j in range(K):
            for i, s in enumerate(s_grid):
                v_interp = np.interp(s_next, s_grid, np.max(V_old, axis=1),
                                   left=np.max(V_old[0]), right=np.max(V_old[-1]))
               
                integl = reward_function(s_next) * transition_density(s_next, s, j+1) + \
                           gamma * v_interp * transition_density(s_next, s, j+1)
               
                F[i, j] = np.trapz(integl, s_next)
       
        V = F
       
        rel_diff = np.max(np.abs((V - V_old) / (np.abs(V_old) + 1e-10)))
        if rel_diff < tol:
            print(f"Converged after {iteration+1} iterations")
            break
           
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, relative diff: {rel_diff:.6f}")
   
    return s_grid, V[:, 0], V[:, 1]
def criterion_C1(pred, target, a=0, b=20):
        z = pred
        phi = (b - a)/(1 + torch.exp(z)) + b * torch.log(1 + torch.exp(z))
        psi = -torch.log(1 + torch.exp(z))
        c1_loss= torch.mean(phi + target * psi)
        
        monitoring_loss = torch.mean((phi - target)**2)  # Monitoring loss
        return c1_loss, monitoring_loss

def criterion_A1(pred, target):
        # Main quadratic loss
        loss = torch.mean(0.5 * pred**2 - target * pred)
       
       
        main_loss = loss 
       
        # Monitoring loss
        monitoring_loss = torch.mean((pred - target)**2)
        return main_loss, monitoring_loss
def train_networks_infinite(data_a1, data_a2, hidden_size=100, epochs=2000, gamma=0.8, criterion="C1"):
    states1 = torch.FloatTensor(data_a1['states'].reshape(-1, 1))
    states2 = torch.FloatTensor(data_a2['states'].reshape(-1, 1))
    next_states1 = torch.FloatTensor(data_a1['next_states'].reshape(-1, 1))
    next_states2 = torch.FloatTensor(data_a2['next_states'].reshape(-1, 1))


    


    if criterion == "C1":
        net1 = ShallowNetwork(hidden_size)
        net2 = ShallowNetwork(hidden_size)
       
        optimizer1 = optim.Adam(net1.parameters(), lr=0.0001)
        optimizer2 = optim.Adam(net2.parameters(), lr=0.0001)
       
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, epochs, eta_min=1e-6)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, epochs, eta_min=1e-6)
       
        def train_step(net, optimizer, states, next_states, rewards):
            with torch.no_grad():
                v1_next = 20 * torch.sigmoid(net1(next_states))
                v2_next = 20 * torch.sigmoid(net2(next_states))
                next_values = torch.maximum(v1_next, v2_next)
                
           
            target = rewards + gamma * next_values
           
            optimizer.zero_grad()
            pred = net(states)
            loss, monitoring_loss = criterion_C1(pred, target)
           
            if torch.isfinite(loss):  
                loss.backward()
                
               
                optimizer.step()
           
            return loss, monitoring_loss
           
    else:  
        net1 = ShallowNetwork(hidden_size)
        net2 = ShallowNetwork(hidden_size)
        optimizer1 = optim.Adam(net1.parameters(), lr=0.0005, weight_decay=0.001)
        optimizer2 = optim.Adam(net2.parameters(), lr=0.0005, weight_decay=0.001)
       
        def train_step(net, optimizer, states, next_states, rewards):
            v1_next = net1(next_states)
            v2_next = net2(next_states)
            next_values = torch.maximum(v1_next.detach(), v2_next.detach())
           
            target = rewards + gamma * next_values
            optimizer.zero_grad()
            pred = net(states)
            loss, monitoring_loss = criterion_A1(pred, target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            return loss, monitoring_loss


    l1, l2 = [], []
    monitoring_l1, monitoring_l2 = [], []
   
    
   
    for epoch in range(epochs):
        rewards1 = torch.FloatTensor(reward_function(data_a1['next_states']).reshape(-1, 1))
        rewards2 = torch.FloatTensor(reward_function(data_a2['next_states']).reshape(-1, 1))
       
        # Train both networks
        loss1, mon_loss1 = train_step(net1, optimizer1, states1, next_states1, rewards1)
        loss2, mon_loss2 = train_step(net2, optimizer2, states2, next_states2, rewards2)
       
        # Update learning rate schedules for C1
        if criterion == "C1":
            scheduler1.step()
            scheduler2.step()
       
        if epoch % 10 == 0:
            l1.append(loss1.item())
            l2.append(loss2.item())
            monitoring_l1.append(mon_loss1.item())
            monitoring_l2.append(mon_loss2.item())
           
            
   
    return net1, net2, (l1, l2), (monitoring_l1, monitoring_l2)
def plot_training_progress(losses_orig, monitoring_losses_orig, losses_A1, monitoring_losses_A1):
    """Plot training progress for both criteria"""
    plt.figure(figsize=(15, 10))
   
    # C1 criterion plots
    plt.subplot(2, 2, 1)
    plt.plot(losses_orig[0], 'r-', label='Action 1')  # No .numpy() needed
    plt.plot(losses_orig[1], 'b-', label='Action 2')
    plt.title('C1 Optimization Losses')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(2, 2, 2)
    plt.plot(monitoring_losses_orig[0], 'r-', label='Action 1')
    plt.plot(monitoring_losses_orig[1], 'b-', label='Action 2')
    plt.title('C1 Mean Squared Error')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
   
    # A1 criterion plots
    plt.subplot(2, 2, 3)
    plt.plot(losses_A1[0], 'r-', label='Action 1')
    plt.plot(losses_A1[1], 'b-', label='Action 2')
    plt.title('A1 Optimization Losses')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(2, 2, 4)
    plt.plot(monitoring_losses_A1[0], label='Action 1', )
    plt.plot(monitoring_losses_A1[1], 'b-', label='Action 2')
    plt.title('A1 Mean Squared Error')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
   
    plt.tight_layout()
    plt.show()




def plot_comparison(s_grid, v1, v2, net1_orig, net2_orig, net1_A1, net2_A1):
    """Plot results comparing both criteria"""
    s_tensor = torch.FloatTensor(s_grid.reshape(-1, 1))
   
    with torch.no_grad():
        # C1 predictions
        c1out = 0*1/(1+torch.exp(net1_orig(s_tensor))) + 20*torch.exp(net1_orig(s_tensor)) / (1+torch.exp(net1_orig(s_tensor)))
        c2out = 0*1/(1+torch.exp(net2_orig(s_tensor))) + 20*torch.exp(net2_orig(s_tensor)) / (1+torch.exp(net2_orig(s_tensor)))
        v1_nn_orig = c1out
        v2_nn_orig = c2out
       
        # A1 predictions
        v1_nn_A1 = net1_A1(s_tensor).numpy()
        v2_nn_A1 = net2_A1(s_tensor).numpy()
   
    plt.figure(figsize=(15, 10))
   
    # Plot C1 results
    plt.subplot(2, 2, 1)
    plt.plot(s_grid, v1, 'r-', label='v₁(s) Numerical')
    plt.plot(s_grid, v1_nn_orig, 'r--', label='v₁(s) C1')
    plt.plot(s_grid, v2, 'b-', label='v₂(s) Numerical')
    plt.plot(s_grid, v2_nn_orig, 'b--', label='v₂(s) C1')
    plt.title('Value Functions (C1)')
    plt.xlabel('State s')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
   
    # Plot A1 results
    plt.subplot(2, 2, 2)
    plt.plot(s_grid, v1, 'r-', label='v₁(s) Numerical')
    plt.plot(s_grid, v1_nn_A1, 'r--', label='v₁(s) A1')
    plt.plot(s_grid, v2, 'b-', label='v₂(s) Numerical')
    plt.plot(s_grid, v2_nn_A1, 'b--', label='v₂(s) A1')
    plt.title('Value Functions (A1)')
    plt.xlabel('State s')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
   
    # Plot C1 policy
    plt.subplot(2, 2, 3)
    optimal_action_num = np.where(v1 >= v2, 1, 2)
    optimal_action_nn_orig = np.where(v1_nn_orig >= v2_nn_orig, 1, 2)
   
    plt.plot(s_grid, optimal_action_num, 'r-', label='Numerical')
    plt.plot(s_grid, optimal_action_nn_orig, 'b--', label='C1')
    plt.title('Optimal Action Policy (C1)')
    plt.xlabel('State s')
    plt.ylabel('Action')
    plt.yticks([1, 2])
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)

    # Plot A1 policy
    plt.subplot(2, 2, 4)
    optimal_action_nn_A1 = np.where(v1_nn_A1 >= v2_nn_A1, 1, 2)
   
    plt.plot(s_grid, optimal_action_num, 'r-', label='Numerical')
    plt.plot(s_grid, optimal_action_nn_A1, 'b--', label='A1')
    plt.title('Optimal Action Policy (A1)')
    plt.xlabel('State s')
    plt.ylabel('Action')
    plt.yticks([1, 2])
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
   
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
   
    # Generate training data
    data_a1, data_a2 = generate_training_data(N=1000)
   
    # Compute numerical solution
    s_grid, v1, v2 = numerical_solution()
   
    # Train networks with C1 criterion
    net1_orig, net2_orig, losses_orig, monitoring_losses_orig = train_networks_infinite(
        data_a1, data_a2, criterion="C1"
    )
   
    # Train networks with A1 criterion
    net1_A1, net2_A1, losses_A1, monitoring_losses_A1 = train_networks_infinite(
        data_a1, data_a2, criterion="A1"
    )
   
    # Plot training progress
    plot_training_progress(losses_orig, monitoring_losses_orig, losses_A1, monitoring_losses_A1)
   
    # Plot comparison
    plot_comparison(s_grid, v1, v2, net1_orig, net2_orig, net1_A1, net2_A1)