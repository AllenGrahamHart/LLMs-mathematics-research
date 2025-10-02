import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg
import seaborn as sns
from sklearn.linear_model import Ridge
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

np.random.seed(42)

# ============================================================================
# Experiment 1: Exploring Input Scaling and Spectral Radius Trade-offs
# ============================================================================

def create_reservoir(N, spectral_radius, density=0.1):
    """Create a random reservoir with specified spectral radius."""
    W = sparse.random(N, N, density=density, data_rvs=np.random.randn)
    eigenvalues = splinalg.eigs(W, k=1, return_eigenvectors=False)
    current_radius = np.abs(eigenvalues[0])
    W = W * (spectral_radius / current_radius)
    return W

def create_input_matrix(N, input_dim, input_scaling):
    """Create input weight matrix."""
    W_in = np.random.uniform(-input_scaling, input_scaling, (N, input_dim))
    return W_in

def run_reservoir(W, W_in, inputs, leak_rate=1.0):
    """Run reservoir with given inputs."""
    N = W.shape[0]
    T = len(inputs)
    states = np.zeros((T, N))
    x = np.zeros(N)
    
    for t in range(T):
        x = (1 - leak_rate) * x + leak_rate * np.tanh(W.dot(x) + W_in.dot(inputs[t]))
        states[t] = x
    
    return states

def lorenz_system(T, dt=0.01, sigma=10, rho=28, beta=8/3):
    """Generate Lorenz attractor data."""
    x, y, z = 1.0, 1.0, 1.0
    data = []
    
    for _ in range(T):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dx*dt, y + dy*dt, z + dz*dt
        data.append([x, y, z])
    
    return np.array(data)

def mackey_glass(T, tau=17, n=10, beta=0.2, gamma=0.1, dt=1):
    """Generate Mackey-Glass time series."""
    history_length = tau + 1
    history = 1.2 + 0.1 * np.random.rand(history_length)
    x = history[-1]
    output = []
    
    for _ in range(T):
        x_tau = history[0]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x
        x = x + dx * dt
        history = np.roll(history, -1)
        history[-1] = x
        output.append(x)
    
    return np.array(output)

# ============================================================================
# Main Experiment: Phase Space Analysis
# ============================================================================

print("=" * 80)
print("RESERVOIR COMPUTING: PHASE SPACE ANALYSIS OF HYPERPARAMETERS")
print("=" * 80)

# Generate tasks
print("\nGenerating benchmark tasks...")
T_train, T_test = 3000, 1000

# Task 1: Lorenz prediction
lorenz_data = lorenz_system(T_train + T_test)
lorenz_data = (lorenz_data - lorenz_data.mean(axis=0)) / lorenz_data.std(axis=0)

# Task 2: Mackey-Glass
mg_data = mackey_glass(T_train + T_test + 100)
mg_data = (mg_data - mg_data.mean()) / mg_data.std()

# Parameter grid
spectral_radii = np.linspace(0.3, 1.5, 10)
input_scalings = np.logspace(-2, 1, 10)
N = 200  # Reservoir size

print(f"\nRunning grid search: {len(spectral_radii)} x {len(input_scalings)} = {len(spectral_radii) * len(input_scalings)} configurations")

# Test on Lorenz prediction
results_lorenz = np.zeros((len(spectral_radii), len(input_scalings)))
results_mg = np.zeros((len(spectral_radii), len(input_scalings)))

for i, sr in enumerate(tqdm(spectral_radii, desc="Spectral radius")):
    W = create_reservoir(N, sr)
    
    for j, input_scale in enumerate(input_scalings):
        W_in = create_input_matrix(N, 3, input_scale)
        
        # Lorenz task
        try:
            states = run_reservoir(W, W_in, lorenz_data[:T_train], leak_rate=0.3)
            X_train, y_train = states[:-1], lorenz_data[1:T_train]
            
            ridge = Ridge(alpha=1e-6)
            ridge.fit(X_train, y_train)
            
            # Test
            states_test = run_reservoir(W, W_in, lorenz_data[T_train:T_train+T_test], leak_rate=0.3)
            y_pred = ridge.predict(states_test[:-1])
            y_true = lorenz_data[T_train+1:T_train+T_test]
            
            mse = np.mean((y_pred - y_true)**2)
            results_lorenz[i, j] = mse
        except:
            results_lorenz[i, j] = np.nan
        
        # Mackey-Glass task
        W_in_mg = create_input_matrix(N, 1, input_scale)
        try:
            mg_input = mg_data[:-1].reshape(-1, 1)
            states = run_reservoir(W, W_in_mg, mg_input[:T_train], leak_rate=0.3)
            
            ridge = Ridge(alpha=1e-6)
            ridge.fit(states, mg_data[1:T_train+1])
            
            states_test = run_reservoir(W, W_in_mg, mg_input[T_train:T_train+T_test], leak_rate=0.3)
            y_pred = ridge.predict(states_test)
            y_true = mg_data[T_train+1:T_train+T_test+1]
            
            mse = np.mean((y_pred - y_true)**2)
            results_mg[i, j] = mse
        except:
            results_mg[i, j] = np.nan

# ============================================================================
# Plotting Results
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lorenz results
im1 = axes[0].imshow(results_lorenz, aspect='auto', origin='lower', 
                     extent=[input_scalings[0], input_scalings[-1], 
                            spectral_radii[0], spectral_radii[-1]],
                     cmap='viridis', vmin=0, vmax=np.nanpercentile(results_lorenz, 95))
axes[0].set_xlabel('Input Scaling', fontsize=12)
axes[0].set_ylabel('Spectral Radius', fontsize=12)
axes[0].set_title('Lorenz Attractor Prediction (MSE)', fontsize=13, fontweight='bold')
axes[0].set_xscale('log')
plt.colorbar(im1, ax=axes[0], label='MSE')

# Mackey-Glass results
im2 = axes[1].imshow(results_mg, aspect='auto', origin='lower',
                     extent=[input_scalings[0], input_scalings[-1],
                            spectral_radii[0], spectral_radii[-1]],
                     cmap='viridis', vmin=0, vmax=np.nanpercentile(results_mg, 95))
axes[1].set_xlabel('Input Scaling', fontsize=12)
axes[1].set_ylabel('Spectral Radius', fontsize=12)
axes[1].set_title('Mackey-Glass Prediction (MSE)', fontsize=13, fontweight='bold')
axes[1].set_xscale('log')
plt.colorbar(im2, ax=axes[1], label='MSE')

plt.tight_layout()
plt.savefig('outputs/open_research_20251002_181646/phase_space.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved phase_space.png")

# ============================================================================
# Experiment 2: Edge of Chaos Analysis
# ============================================================================

print("\n" + "=" * 80)
print("EDGE OF CHAOS ANALYSIS")
print("=" * 80)

def compute_lyapunov_exponent(W, W_in, input_data, n_iterations=1000, epsilon=1e-8):
    """Estimate largest Lyapunov exponent of reservoir dynamics."""
    N = W.shape[0]
    x = np.random.randn(N) * 0.1
    
    lyap_sum = 0
    count = 0
    
    for t in range(min(n_iterations, len(input_data))):
        # Perturbed state
        delta = np.random.randn(N)
        delta = delta / np.linalg.norm(delta) * epsilon
        x_pert = x + delta
        
        # Evolve both states
        u = input_data[t]
        x_new = np.tanh(W.dot(x) + W_in.dot(u))
        x_pert_new = np.tanh(W.dot(x_pert) + W_in.dot(u))
        
        # Compute divergence
        divergence = np.linalg.norm(x_new - x_pert_new)
        if divergence > 0:
            lyap_sum += np.log(divergence / epsilon)
            count += 1
        
        x = x_new
    
    return lyap_sum / count if count > 0 else 0

spectral_radii_eoc = np.linspace(0.1, 2.0, 20)
lyapunov_exponents = []
task_performance = []

print("\nComputing Lyapunov exponents and task performance...")

for sr in tqdm(spectral_radii_eoc):
    W = create_reservoir(N, sr, density=0.1)
    W_in = create_input_matrix(N, 1, 0.5)
    
    # Compute Lyapunov exponent
    test_input = np.random.randn(1000, 1) * 0.1
    lyap = compute_lyapunov_exponent(W, W_in, test_input, n_iterations=500)
    lyapunov_exponents.append(lyap)
    
    # Test performance on Mackey-Glass
    try:
        mg_input = mg_data[:-1].reshape(-1, 1)
        states = run_reservoir(W, W_in, mg_input[:T_train], leak_rate=0.3)
        ridge = Ridge(alpha=1e-5)
        ridge.fit(states, mg_data[1:T_train+1])
        
        states_test = run_reservoir(W, W_in, mg_input[T_train:T_train+500], leak_rate=0.3)
        y_pred = ridge.predict(states_test)
        y_true = mg_data[T_train+1:T_train+501]
        mse = np.mean((y_pred - y_true)**2)
        task_performance.append(mse)
    except:
        task_performance.append(np.nan)

# Plot edge of chaos analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(spectral_radii_eoc, lyapunov_exponents, 'o-', linewidth=2, markersize=6)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Lyapunov = 0')
ax1.set_xlabel('Spectral Radius', fontsize=12)
ax1.set_ylabel('Lyapunov Exponent', fontsize=12)
ax1.set_title('Reservoir Dynamics: Lyapunov Exponent vs Spectral Radius', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(spectral_radii_eoc, task_performance, 's-', linewidth=2, markersize=6, color='green')
ax2.set_xlabel('Spectral Radius', fontsize=12)
ax2.set_ylabel('Task MSE', fontsize=12)
ax2.set_title('Task Performance vs Spectral Radius', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('outputs/open_research_20251002_181646/edge_of_chaos.png', dpi=300, bbox_inches='tight')
print("✓ Saved edge_of_chaos.png")

# ============================================================================
# Experiment 3: Information Processing Capacity
# ============================================================================

print("\n" + "=" * 80)
print("INFORMATION PROCESSING CAPACITY")
print("=" * 80)

def compute_memory_capacity(W, W_in, max_delay=50, T_test=2000):
    """Compute memory capacity of reservoir."""
    input_signal = np.random.uniform(-1, 1, T_test).reshape(-1, 1)
    states = run_reservoir(W, W_in, input_signal, leak_rate=0.3)
    
    capacities = []
    for k in range(1, max_delay + 1):
        if k >= T_test:
            break
        
        X = states[k:]
        y = input_signal[:-k]
        
        if len(X) < 100:
            break
        
        ridge = Ridge(alpha=1e-5)
        ridge.fit(X, y)
        y_pred = ridge.predict(X)
        
        # Compute correlation coefficient squared (R^2)
        correlation = np.corrcoef(y.flatten(), y_pred.flatten())[0, 1]
        capacity = correlation ** 2 if not np.isnan(correlation) else 0
        capacities.append(capacity)
    
    return np.array(capacities)

configs = [
    (0.5, 0.5, 'Low SR, Low Input'),
    (0.9, 0.5, 'Optimal SR, Low Input'),
    (1.3, 0.5, 'High SR, Low Input'),
    (0.9, 0.1, 'Optimal SR, Very Low Input'),
    (0.9, 2.0, 'Optimal SR, High Input'),
]

plt.figure(figsize=(12, 6))

for sr, inp_scale, label in configs:
    W = create_reservoir(N, sr)
    W_in = create_input_matrix(N, 1, inp_scale)
    mc = compute_memory_capacity(W, W_in, max_delay=30)
    plt.plot(range(1, len(mc) + 1), mc, 'o-', label=label, linewidth=2, markersize=4, alpha=0.8)

plt.xlabel('Delay (k)', fontsize=12)
plt.ylabel('Memory Capacity $MC_k$', fontsize=12)
plt.title('Memory Capacity for Different Hyperparameter Configurations', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/open_research_20251002_181646/memory_capacity.png', dpi=300, bbox_inches='tight')
print("✓ Saved memory_capacity.png")

print("\n" + "=" * 80)
print("EXPERIMENTS COMPLETE")
print("=" * 80)
