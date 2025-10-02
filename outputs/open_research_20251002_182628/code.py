import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("=" * 60)
print("FINAL ITERATION: Completing Paper with Existing Results")
print("=" * 60)

# Implement Echo State Network (without scipy dependencies)
class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, 
                 spectral_radius=0.9, sparsity=0.1, input_scaling=1.0,
                 leak_rate=1.0, random_state=42):
        np.random.seed(random_state)
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.leak_rate = leak_rate
        
        self.W_in = np.random.uniform(-input_scaling, input_scaling, 
                                      (n_reservoir, n_inputs))
        
        W = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) < sparsity
        W = W * mask
        
        eigenvalues = np.linalg.eigvals(W)
        W = W * (spectral_radius / np.max(np.abs(eigenvalues)))
        self.W = W
        
        self.W_out = None
        self.last_state = np.zeros(n_reservoir)
        
    def _update(self, state, input_pattern):
        pre_activation = np.dot(self.W, state) + np.dot(self.W_in, input_pattern)
        new_state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
        return new_state
    
    def fit(self, X, y, washout=100, ridge_alpha=1e-6):
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        
        state = self.last_state
        for t in range(n_samples):
            state = self._update(state, X[t])
            states[t] = state
        
        states_train = states[washout:]
        y_train = y[washout:]
        
        self.W_out = np.dot(
            np.linalg.inv(np.dot(states_train.T, states_train) + 
                         ridge_alpha * np.eye(self.n_reservoir)),
            np.dot(states_train.T, y_train)
        )
        
        self.last_state = state
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_outputs))
        
        state = self.last_state
        for t in range(n_samples):
            state = self._update(state, X[t])
            predictions[t] = np.dot(self.W_out.T, state)
        
        self.last_state = state
        return predictions

# Generate time series using simple RK4 for Lorenz
def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + dt/2 * k1, t + dt/2)
    k3 = f(y + dt/2 * k2, t + dt/2)
    k4 = f(y + dt * k3, t + dt)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz(n_points=5000, dt=0.01):
    def lorenz_derivs(state, t, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    
    states = np.zeros((n_points, 3))
    states[0] = [1.0, 1.0, 1.0]
    
    for i in range(1, n_points):
        states[i] = rk4_step(lorenz_derivs, states[i-1], i*dt, dt)
    
    return states[:, 0]

def generate_periodic(n_points=5000, dt=0.01):
    t = np.arange(0, n_points * dt, dt)
    return np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)

def generate_noisy_periodic(n_points=5000, dt=0.01, noise_level=0.3):
    return generate_periodic(n_points, dt) + np.random.randn(n_points) * noise_level

# Main experiments
print("\nRunning sparsity-performance analysis...")
sparsity_levels = np.linspace(0.05, 0.5, 10)
n_trials = 5
regimes = {
    'Chaotic (Lorenz)': generate_lorenz,
    'Periodic': generate_periodic,
    'Noisy Periodic': generate_noisy_periodic
}

results = {regime: {'mean': [], 'std': []} for regime in regimes}

for regime_name, generator in regimes.items():
    print(f"  Testing {regime_name}...")
    
    for sparsity in sparsity_levels:
        errors = []
        
        for trial in range(n_trials):
            data = generator(n_points=3000)
            data = (data - np.mean(data)) / np.std(data)
            
            n_train = 2000
            X_train = data[:n_train].reshape(-1, 1)
            y_train = data[1:n_train+1].reshape(-1, 1)
            X_test = data[n_train:n_train+500].reshape(-1, 1)
            y_test = data[n_train+1:n_train+501].reshape(-1, 1)
            
            esn = EchoStateNetwork(
                n_inputs=1, n_reservoir=200, n_outputs=1,
                spectral_radius=0.9, sparsity=sparsity,
                random_state=trial
            )
            esn.fit(X_train, y_train, washout=100)
            y_pred = esn.predict(X_test)
            
            errors.append(mean_squared_error(y_test, y_pred))
        
        results[regime_name]['mean'].append(np.mean(errors))
        results[regime_name]['std'].append(np.std(errors))

# Figure 1: Performance vs Sparsity
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71']

for (regime_name, result), color in zip(results.items(), colors):
    mean_errors = result['mean']
    std_errors = result['std']
    ax.plot(sparsity_levels, mean_errors, 'o-', label=regime_name, 
            color=color, linewidth=2, markersize=6)
    ax.fill_between(sparsity_levels, 
                     np.array(mean_errors) - np.array(std_errors),
                     np.array(mean_errors) + np.array(std_errors),
                     alpha=0.2, color=color)

ax.set_xlabel('Reservoir Sparsity', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('Prediction Performance vs Reservoir Sparsity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/sparsity_performance.png", dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved")

# Figure 2: Eigenvalue spectra
optimal_sparsities = {
    'Chaotic (Lorenz)': sparsity_levels[np.argmin(results['Chaotic (Lorenz)']['mean'])],
    'Periodic': sparsity_levels[np.argmin(results['Periodic']['mean'])],
    'Noisy Periodic': sparsity_levels[np.argmin(results['Noisy Periodic']['mean'])]
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (regime_name, opt_sparsity) in enumerate(optimal_sparsities.items()):
    esn = EchoStateNetwork(n_inputs=1, n_reservoir=200, n_outputs=1,
                          spectral_radius=0.9, sparsity=opt_sparsity, random_state=42)
    eigenvalues = np.linalg.eigvals(esn.W)
    
    axes[idx].scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6, s=20, color=colors[idx])
    circle = plt.Circle((0, 0), 0.9, fill=False, color='red', linestyle='--', linewidth=2)
    axes[idx].add_patch(circle)
    axes[idx].set_xlabel('Real', fontsize=11)
    axes[idx].set_ylabel('Imaginary', fontsize=11)
    axes[idx].set_title(f'{regime_name}\n(sparsity={opt_sparsity:.2f})', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_aspect('equal')
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{output_dir}/eigenvalue_spectra.png", dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved")

# Figure 3: Memory capacity
sparsity_range = np.linspace(0.05, 0.5, 8)
mc_values = []

for sparsity in sparsity_range:
    esn = EchoStateNetwork(n_inputs=1, n_reservoir=200, n_outputs=1,
                          spectral_radius=0.95, sparsity=sparsity, random_state=42)
    
    u = np.random.uniform(-0.5, 0.5, 1500)
    states = np.zeros((1500, esn.n_reservoir))
    state = np.zeros(esn.n_reservoir)
    
    for t in range(1500):
        state = esn._update(state, np.array([u[t]]))
        states[t] = state
    
    mc_total = 0
    washout = 100
    states_train = states[washout:]
    
    for k in range(1, 31):
        target = u[washout - k:-k]
        states_k = states_train[:len(target)]
        
        W_out = np.dot(np.linalg.pinv(states_k), target)
        y_pred = np.dot(states_k, W_out)
        mc_k = np.corrcoef(target, y_pred)[0, 1] ** 2
        mc_total += max(0, mc_k)
    
    mc_values.append(mc_total)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sparsity_range, mc_values, 'o-', color='#9b59b6', linewidth=2, markersize=8)
ax.set_xlabel('Reservoir Sparsity', fontsize=12)
ax.set_ylabel('Memory Capacity', fontsize=12)
ax.set_title('Memory Capacity vs Reservoir Sparsity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/memory_capacity.png", dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved")

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETED")
print("="*60)
