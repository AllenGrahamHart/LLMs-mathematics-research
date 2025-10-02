import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("FINAL ITERATION: GENERATING ALL FIGURES")
print("=" * 60)

# Implement ESN 
class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, 
                 spectral_radius=0.9, sparsity=0.1, 
                 input_scaling=1.0, ridge_alpha=1e-6,
                 W_reservoir=None):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
        
        # Initialize input weights
        self.W_in = np.random.randn(n_reservoir, n_inputs) * input_scaling
        
        # Create reservoir weights (or use provided)
        if W_reservoir is None:
            self.W = self._create_random_reservoir()
        else:
            self.W = W_reservoir
            
        # Scale to desired spectral radius
        self._scale_spectral_radius()
        
        self.W_out = None
        self.state = np.zeros(n_reservoir)
        
    def _create_random_reservoir(self):
        """Create random sparse reservoir"""
        W = sparse.random(self.n_reservoir, self.n_reservoir, 
                         density=self.sparsity, format='csr')
        return W.toarray()
    
    def _scale_spectral_radius(self):
        """Scale reservoir to desired spectral radius"""
        eigenvalues = np.linalg.eigvals(self.W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            self.W *= self.spectral_radius / current_radius
        
    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)
        
    def update(self, input_signal):
        """Update reservoir state"""
        self.state = np.tanh(
            np.dot(self.W, self.state) + np.dot(self.W_in, input_signal)
        )
        return self.state
    
    def fit(self, X, Y, washout=100):
        """Train output weights using ridge regression"""
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        
        self.reset_state()
        for t in range(n_samples):
            states[t] = self.update(X[t])
        
        # Remove washout period
        states_train = states[washout:]
        Y_train = Y[washout:]
        
        # Ridge regression
        self.W_out = np.linalg.solve(
            states_train.T @ states_train + 
            self.ridge_alpha * np.eye(self.n_reservoir),
            states_train.T @ Y_train
        ).T
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_outputs))
        
        self.reset_state()
        for t in range(n_samples):
            state = self.update(X[t])
            Y_pred[t] = np.dot(self.W_out, state)
        
        return Y_pred


def create_modular_reservoir(n_reservoir, n_modules, intra_density, inter_density):
    """Create a modular reservoir with specified connectivity"""
    W = np.zeros((n_reservoir, n_reservoir))
    module_size = n_reservoir // n_modules
    
    for i in range(n_modules):
        start = i * module_size
        end = (i + 1) * module_size if i < n_modules - 1 else n_reservoir
        
        # Intra-module connections
        mask = np.random.rand(end - start, end - start) < intra_density
        W[start:end, start:end] = np.random.randn(end - start, end - start) * mask
        
        # Inter-module connections
        for j in range(n_modules):
            if i != j:
                start_j = j * module_size
                end_j = (j + 1) * module_size if j < n_modules - 1 else n_reservoir
                mask = np.random.rand(end - start, end_j - start_j) < inter_density
                W[start:end, start_j:end_j] = np.random.randn(end - start, end_j - start_j) * mask
    
    return W


def generate_memory_task(n_samples, delay):
    """Generate data for k-delay memory task"""
    u = np.random.uniform(-0.5, 0.5, n_samples)
    y = np.zeros(n_samples)
    y[delay:] = u[:-delay]
    return u.reshape(-1, 1), y.reshape(-1, 1)


def compute_memory_capacity(esn, max_delay=50, n_samples=2000, washout=100):
    """Compute total memory capacity across multiple delays"""
    mc_values = []
    
    for k in range(1, max_delay + 1):
        X, Y = generate_memory_task(n_samples, k)
        
        # Split into train and test
        split = n_samples // 2
        esn.fit(X[:split], Y[:split], washout=washout)
        Y_pred = esn.predict(X[split:])
        
        # Compute correlation-based memory capacity
        var_y = np.var(Y[split:])
        mse = np.mean((Y[split:] - Y_pred) ** 2)
        mc_k = max(0, 1 - mse / var_y) if var_y > 0 else 0
        mc_values.append(mc_k)
        
        if k > 10 and mc_k < 0.01:
            mc_values.extend([0] * (max_delay - k))
            break
    
    return np.array(mc_values), np.sum(mc_values)


def compute_modular_densities(n_modules, target_density, modularity_ratio=3):
    """Compute intra and inter densities to match target overall density"""
    p_inter = target_density * n_modules / (modularity_ratio + n_modules - 1)
    p_intra = modularity_ratio * p_inter
    return p_intra, p_inter


# Experimental parameters
n_reservoir = 200
target_density = 0.1
n_trials = 5

print("\nRunning experiments...")

# Random baseline
random_mc_totals = []
random_mc_curves = []
for trial in range(n_trials):
    esn = EchoStateNetwork(n_inputs=1, n_reservoir=n_reservoir, 
                          n_outputs=1, spectral_radius=0.9,
                          sparsity=target_density)
    mc_values, mc_total = compute_memory_capacity(esn, max_delay=50)
    random_mc_totals.append(mc_total)
    random_mc_curves.append(mc_values)

random_mc_mean = np.mean(random_mc_totals)
random_mc_std = np.std(random_mc_totals)
random_mc_curve_mean = np.mean(random_mc_curves, axis=0)

# Modular configurations
modularity_configs = [(2, 3.0), (4, 3.0), (5, 3.0), (10, 3.0), (20, 3.0)]
modular_results = []
modular_curves = {}

for n_modules, gamma in modularity_configs:
    p_intra, p_inter = compute_modular_densities(n_modules, target_density, gamma)
    mc_totals = []
    mc_curves_list = []
    
    for trial in range(n_trials):
        W = create_modular_reservoir(n_reservoir, n_modules, p_intra, p_inter)
        esn = EchoStateNetwork(n_inputs=1, n_reservoir=n_reservoir, 
                              n_outputs=1, spectral_radius=0.9,
                              W_reservoir=W)
        mc_values, mc_total = compute_memory_capacity(esn, max_delay=50)
        mc_totals.append(mc_total)
        mc_curves_list.append(mc_values)
    
    mc_mean = np.mean(mc_totals)
    mc_std = np.std(mc_totals)
    mc_curve_mean = np.mean(mc_curves_list, axis=0)
    
    modular_results.append((n_modules, mc_mean, mc_std))
    modular_curves[n_modules] = mc_curve_mean

print("✓ Experiments complete\n")
print("Generating figures...")

# Figure 1: Memory capacity vs modularity
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
n_modules_list = [r[0] for r in modular_results]
mc_means = [r[1] for r in modular_results]
mc_stds = [r[2] for r in modular_results]

ax.errorbar(n_modules_list, mc_means, yerr=mc_stds, 
           marker='o', linewidth=2, markersize=8, capsize=5, 
           label='Modular', color='steelblue')
ax.axhline(y=random_mc_mean, color='red', linestyle='--', 
          linewidth=2, label='Random (baseline)')
ax.fill_between([0, max(n_modules_list) + 2], 
                random_mc_mean - random_mc_std,
                random_mc_mean + random_mc_std,
                color='red', alpha=0.2)

ax.set_xlabel('Number of Modules', fontsize=12)
ax.set_ylabel('Total Memory Capacity', fontsize=12)
ax.set_title('Memory Capacity vs Network Modularity\n(Equal Total Connectivity Density)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(n_modules_list) + 2)
plt.tight_layout()
plt.savefig("memory_vs_modularity.png", dpi=300)
print("✓ Saved: memory_vs_modularity.png")
plt.close()

# Figure 2: MC decay curves
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
delays = np.arange(1, 51)
ax.plot(delays, random_mc_curve_mean, linewidth=2.5, label='Random', 
       color='red', alpha=0.8)
ax.plot(delays, modular_curves[5], linewidth=2.5, 
       label='Modular (5 modules)', color='green', alpha=0.8)
ax.plot(delays, modular_curves[20], linewidth=2.5, 
       label='Modular (20 modules)', color='orange', alpha=0.8, linestyle='--')

ax.set_xlabel('Delay (k)', fontsize=12)
ax.set_ylabel('Memory Capacity $MC_k$', fontsize=12)
ax.set_title('Memory Capacity Decay Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 50)
plt.tight_layout()
plt.savefig("mc_decay_curves.png", dpi=300)
print("✓ Saved: mc_decay_curves.png")
plt.close()

# Figure 3: Eigenvalue spectra
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
configs = [
    ('Random', None, 1, 'blue'),
    ('Modular (5 modules)', 5, 3.0, 'green'),
    ('Modular (20 modules)', 20, 3.0, 'orange')
]

for idx, (title, n_mod, gamma, color) in enumerate(configs):
    if n_mod is None:
        W = sparse.random(n_reservoir, n_reservoir, density=target_density).toarray()
    else:
        p_intra, p_inter = compute_modular_densities(n_mod, target_density, gamma)
        W = create_modular_reservoir(n_reservoir, n_mod, p_intra, p_inter)
    
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 0:
        eigenvalues *= 0.9 / current_radius
    
    axes[idx].scatter(eigenvalues.real, eigenvalues.imag, alpha=0.5, s=15, color=color)
    axes[idx].add_patch(plt.Circle((0, 0), 0.9, fill=False, 
                                  color='red', linestyle='--', linewidth=2))
    axes[idx].set_xlabel('Real', fontsize=11)
    axes[idx].set_ylabel('Imaginary', fontsize=11)
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(-1.1, 1.1)
    axes[idx].set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("eigenvalue_spectra.png", dpi=300)
print("✓ Saved: eigenvalue_spectra.png")
plt.close()

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nRandom baseline: MC = {random_mc_mean:.2f} ± {random_mc_std:.2f}")
print("\nModular configurations:")
for n_mod, mc_mean, mc_std in modular_results:
    improvement = ((mc_mean / random_mc_mean) - 1) * 100
    print(f"  {n_mod:2d} modules: MC = {mc_mean:.2f} ± {mc_std:.2f} ({improvement:+.1f}%)")
