import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class ReservoirComputer:
    def __init__(self, n_reservoir: int, spectral_radius: float = 0.9, 
                 input_scaling: float = 1.0, leak_rate: float = 1.0):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.W_reservoir = self._initialize_reservoir()
        self.W_in = None
        self.W_out = None
        self.state = np.zeros(n_reservoir)
        
    def _initialize_reservoir(self) -> np.ndarray:
        W = np.random.randn(self.n_reservoir, self.n_reservoir) * 0.5
        W[np.random.rand(*W.shape) > 0.1] = 0
        eigenvalues = np.linalg.eigvals(W)
        W = W / (np.max(np.abs(eigenvalues)) / self.spectral_radius)
        return W
    
    def initialize_input(self, input_dim: int):
        self.W_in = (np.random.rand(self.n_reservoir, input_dim) - 0.5) * self.input_scaling
        
    def update(self, u: np.ndarray) -> np.ndarray:
        pre_activation = self.W_reservoir @ self.state + self.W_in @ u
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre_activation)
        return self.state
    
    def train(self, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
        states = []
        self.state = np.zeros(self.n_reservoir)
        for u in U:
            states.append(self.update(u))
        X = np.array(states)
        self.W_out = Y.T @ X @ np.linalg.inv(X.T @ X + ridge * np.eye(self.n_reservoir))
        
    def predict(self, U: np.ndarray) -> np.ndarray:
        predictions = []
        self.state = np.zeros(self.n_reservoir)
        for u in U:
            predictions.append(self.W_out @ self.update(u))
        return np.array(predictions)

def compute_memory_capacity(reservoir: ReservoirComputer, n_delays: int = 50, 
                           n_train: int = 2000, n_test: int = 1000) -> Tuple[np.ndarray, float]:
    u_train = np.random.uniform(-1, 1, (n_train, 1))
    u_test = np.random.uniform(-1, 1, (n_test, 1))
    capacities = []
    
    for k in range(1, n_delays + 1):
        if k < n_train and k < n_test:
            y_train = u_train[:-k]
            u_train_k = u_train[k:]
            reservoir.initialize_input(1)
            reservoir.train(u_train_k, y_train)
            y_test = u_test[:-k]
            u_test_k = u_test[k:]
            y_pred = reservoir.predict(u_test_k)
            cov = np.cov(y_test.flatten(), y_pred.flatten())[0, 1]
            var_target = np.var(y_test)
            var_pred = np.var(y_pred)
            if var_target > 0 and var_pred > 0:
                mc_k = (cov ** 2) / (var_target * var_pred)
                capacities.append(max(0, mc_k))
            else:
                capacities.append(0)
        else:
            capacities.append(0)
    return np.array(capacities), np.sum(capacities)

def compute_kernel_quality(reservoir: ReservoirComputer, n_samples: int = 500) -> float:
    U = np.random.uniform(-1, 1, (n_samples, 1))
    reservoir.initialize_input(1)
    states = []
    reservoir.state = np.zeros(reservoir.n_reservoir)
    for u in U:
        states.append(reservoir.update(u))
    X = np.array(states)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S_normalized = S / np.sum(S)
    entropy_val = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
    effective_rank = np.exp(entropy_val)
    return effective_rank / len(S)

# Experiments
print("Running experiments...")
n_reservoir = 100
spectral_radii = np.linspace(0.1, 1.5, 15)
memory_capacities, kernel_qualities = [], []

for sr in spectral_radii:
    rc = ReservoirComputer(n_reservoir=n_reservoir, spectral_radius=sr)
    _, mc = compute_memory_capacity(rc, n_delays=30, n_train=1000, n_test=500)
    kq = compute_kernel_quality(rc, n_samples=300)
    memory_capacities.append(mc)
    kernel_qualities.append(kq)

network_sizes = [20, 50, 100, 200, 400]
size_memory, size_kernel = [], []
for n in network_sizes:
    rc = ReservoirComputer(n_reservoir=n, spectral_radius=0.9)
    _, mc = compute_memory_capacity(rc, n_delays=min(30, n), n_train=1000, n_test=500)
    kq = compute_kernel_quality(rc, n_samples=300)
    size_memory.append(mc)
    size_kernel.append(kq)

leak_rates = np.linspace(0.1, 1.0, 10)
leak_memory, leak_kernel = [], []
for lr in leak_rates:
    rc = ReservoirComputer(n_reservoir=100, spectral_radius=0.9, leak_rate=lr)
    _, mc = compute_memory_capacity(rc, n_delays=30, n_train=1000, n_test=500)
    kq = compute_kernel_quality(rc, n_samples=300)
    leak_memory.append(mc)
    leak_kernel.append(kq)

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(spectral_radii, memory_capacities, 'o-', linewidth=2, markersize=6)
axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Spectral Radius', fontsize=11)
axes[0, 0].set_ylabel('Memory Capacity', fontsize=11)
axes[0, 0].set_title('Memory Capacity vs Spectral Radius', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(spectral_radii, kernel_qualities, 'o-', color='green', linewidth=2, markersize=6)
axes[0, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Spectral Radius', fontsize=11)
axes[0, 1].set_ylabel('Kernel Quality', fontsize=11)
axes[0, 1].set_title('Kernel Quality vs Spectral Radius', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(memory_capacities, kernel_qualities, c=spectral_radii, 
                   cmap='viridis', s=100, edgecolors='black', linewidth=1)
axes[0, 2].set_xlabel('Memory Capacity', fontsize=11)
axes[0, 2].set_ylabel('Kernel Quality', fontsize=11)
axes[0, 2].set_title('Memory-Kernel Tradeoff', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], label='Spectral Radius')

axes[1, 0].plot(network_sizes, size_memory, 'o-', linewidth=2, markersize=6)
ax2 = axes[1, 0].twinx()
ax2.plot(network_sizes, size_kernel, 's-', color='orange', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Network Size', fontsize=11)
axes[1, 0].set_ylabel('Memory Capacity', fontsize=11)
ax2.set_ylabel('Kernel Quality', fontsize=11, color='orange')
axes[1, 0].set_title('Scaling with Network Size', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(leak_rates, leak_memory, 'o-', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Leak Rate', fontsize=11)
axes[1, 1].set_ylabel('Memory Capacity', fontsize=11)
axes[1, 1].set_title('Memory vs Leak Rate', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(leak_rates, leak_kernel, 'o-', color='green', linewidth=2, markersize=6)
axes[1, 2].set_xlabel('Leak Rate', fontsize=11)
axes[1, 2].set_ylabel('Kernel Quality', fontsize=11)
axes[1, 2].set_title('Kernel Quality vs Leak Rate', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reservoir_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: reservoir_analysis.png")
