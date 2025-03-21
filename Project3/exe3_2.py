import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import integrate
# Loss Functions
def criterion_C1(pred, target, a=0, b=2):
    z = pred
    phi = (b - a) / (1 + torch.exp(z)) + b * torch.log(1 + torch.exp(z))
    psi = -torch.log(1 + torch.exp(z))
    return torch.mean(phi + target * psi)

def criterion_A1(pred, target):
  
    return torch.mean(0.5 * pred**2 - target * pred)

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

#Rewrard Function 
def reward_function(s):
   
    return np.minimum(2, s**2)
#Transition Density 
def transition_density(s_next, s_val, act):
  
    if act == 1:
        mean = 0.8 * s_val + 1.0
    else:
        mean = -2.0
    
    sigma = 1.0
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((s_next - mean) / sigma) ** 2
    )

# Data Generation
def generate_training_data(N=1000):
    st = np.zeros(N + 1)
    st_next = np.zeros(N)
    acts = np.random.choice([1, 2], size=N)
    
    st[0] = np.random.standard_normal()
    
    for t in range(N):
        if acts[t] == 1:
            st_next[t] = 0.8 * st[t] + 1.0 + np.random.standard_normal()
        else:
            st_next[t] = -2.0 + np.random.standard_normal()
        
        if t < N:
            st[t + 1] = st_next[t]
    
    mask1 = acts == 1
    mask2 = acts == 2
    
    data_a1 = {
        'states': st[:-1][mask1],
        'next_states': st_next[mask1]
    }
    data_a2 = {
        'states': st[:-1][mask2],
        'next_states': st_next[mask2]
    }
    
    return data_a1, data_a2

# Numerical Solution
def numerical_solution(grid_points=1000):
    svals = np.linspace(-20, 20, grid_points)
    ds = svals[1] - svals[0]
    
    v1_arr = np.zeros(grid_points)
    v2_arr = np.zeros(grid_points)
    
    for i, sv in enumerate(svals):
        integrand1 = lambda sn: reward_function(sn) * transition_density(sn, sv, 1)
        integrand2 = lambda sn: reward_function(sn) * transition_density(sn, sv, 2)
        
        v1_arr[i] = np.trapz(integrand1(svals), svals)
        v2_arr[i] = np.trapz(integrand2(svals), svals)
    
    return svals, v1_arr, v2_arr

# Training
def train_networks(data_a1, data_a2, hidden_size=100, epochs=2000, criterion_type="C1"):
   
    states1 = torch.FloatTensor(data_a1['states'].reshape(-1, 1))
    states2 = torch.FloatTensor(data_a2['states'].reshape(-1, 1))
    
    rewards1 = torch.FloatTensor(reward_function(data_a1['next_states']).reshape(-1, 1))
    rewards2 = torch.FloatTensor(reward_function(data_a2['next_states']).reshape(-1, 1))
    
    net1 = ShallowNetwork(hidden_size)
    net2 = ShallowNetwork(hidden_size)
    
    optim1 = optim.Adam(net1.parameters(), lr=0.001)
    optim2 = optim.Adam(net2.parameters(), lr=0.001)
    
    loss_func = criterion_C1 if criterion_type == "C1" else criterion_A1
    
    loss_list1, loss_list2 = [], []
    
    for e in range(epochs):
        # Action 1  training
        optim1.zero_grad()
        p1 = net1(states1)
        l1 = loss_func(p1, rewards1)
        l1.backward()
        optim1.step()
        
        # Action 2  training
        optim2.zero_grad()
        p2 = net2(states2)
        l2 = loss_func(p2, rewards2)
        l2.backward()
        optim2.step()
        
        # Log  every 100 epochs
        if e % 100 == 0:
            loss_list1.append(l1.item())
            loss_list2.append(l2.item())
            print(f'Epoch {e}/{epochs} ({criterion_type})')
            print(f'Action 1 - Loss: {l1.item():.4f}')
            print(f'Action 2 - Loss: {l2.item():.4f}')
            print('-' * 50)
    
    return net1, net2, (loss_list1, loss_list2)

def plot_loss(losses_C1, losses_A1):
   
    plt.figure(figsize=(10, 5))
    
    # C1 losses
    plt.subplot(1, 2, 1)
    plt.plot(losses_C1[0], 'r-', label='Action 1')
    plt.plot(losses_C1[1], 'b-', label='Action 2')
    plt.title('C1 Optimization Losses')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # A1 losses
    plt.subplot(1, 2, 2)
    plt.plot(losses_A1[0], 'r-', label='Action 1')
    plt.plot(losses_A1[1], 'b-', label='Action 2')
    plt.title('A1 Optimization Losses')
    plt.xlabel('Epochs (×100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(svals, v1_arr, v2_arr, net1_C1, net2_C1, net1_A1, net2_A1):
    
    s_tensor = torch.FloatTensor(svals.reshape(-1, 1))
    
    with torch.no_grad():
        # C1
        v1_c1 = 2 * torch.sigmoid(net1_C1(s_tensor)).numpy()
        v2_c1 = 2 * torch.sigmoid(net2_C1(s_tensor)).numpy()
        
        # A1
        v1_a1 = net1_A1(s_tensor).numpy()
        v2_a1 = net2_A1(s_tensor).numpy()
    
    plt.figure(figsize=(15, 10))
    
#####   
    plt.subplot(2, 2, 1)
    plt.plot(svals, v1_arr, 'r-', label='v₁(s) Numerical')
    plt.plot(svals, v1_c1, 'r--', label='v₁(s) C1')
    plt.plot(svals, v2_arr, 'b-', label='v₂(s) Numerical')
    plt.plot(svals, v2_c1, 'b--', label='v₂(s) C1')
    plt.title('Value Functions (C1)')
    plt.xlabel('State s')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    #####
    plt.subplot(2, 2, 2)
    plt.plot(svals, v1_arr, 'r-', label='v₁(s) Numerical')
    plt.plot(svals, v1_a1, 'r--', label='v₁(s) A1')
    plt.plot(svals, v2_arr, 'b-', label='v₂(s) Numerical')
    plt.plot(svals, v2_a1, 'b--', label='v₂(s) A1')
    plt.title('Value Functions (A1)')
    plt.xlabel('State s')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    ##########
    plt.subplot(2, 2, 3)
    opt_act_num = np.where(v1_arr >= v2_arr, 1, 2)
    opt_act_c1 = np.where(v1_c1 >= v2_c1, 1, 2)
    plt.plot(svals, opt_act_num, 'r-', label='Numerical')
    plt.plot(svals, opt_act_c1, 'b--', label='C1')
    plt.title('Optimal Action Policy (C1)')
    plt.xlabel('State s')
    plt.ylabel('Action')
    plt.yticks([1, 2])
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    ##################
    plt.subplot(2, 2, 4)
    opt_act_a1 = np.where(v1_a1 >= v2_a1, 1, 2)
    plt.plot(svals, opt_act_num, 'r-', label='Numerical')
    plt.plot(svals, opt_act_a1, 'b--', label='A1')
    plt.title('Optimal Action Policy (A1)')
    plt.xlabel('State s')
    plt.ylabel('Action')
    plt.yticks([1, 2])
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    
    plt.tight_layout()
    plt.show()

def plot_reward_function(rew_func, s_min=-5, s_max=5, n_points=100):
    
    svals = np.linspace(s_min, s_max, n_points)
    rewards = rew_func(svals)
    
    plt.figure(figsize=(6, 4))
    plt.plot(svals, rewards, 'r-', label='R(s)')
    plt.xlabel('State s')
    plt.ylabel('Reward')
    plt.title('Reward Function')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    data1, data2 = generate_training_data(N=1000)
    
    svals, v1_arr, v2_arr = numerical_solution()
    
    net1_C1, net2_C1, losses_C1 = train_networks(data1, data2, criterion_type="C1")
    
    net1_A1, net2_A1, losses_A1 = train_networks(data1, data2, criterion_type="A1")
    
    plot_loss(losses_C1, losses_A1)
    
    plot_comparison(svals, v1_arr, v2_arr, net1_C1, net2_C1, net1_A1, net2_A1)
    
    plot_reward_function(reward_function)