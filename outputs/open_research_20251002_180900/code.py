import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('outputs/open_research_20251002_180900', exist_ok=True)

print("="*60)
print("RESERVOIR COMPUTING: SPECTRAL ANALYSIS OF MEMORY CAPACITY")
print("="*60)

# ============================================================================
# Network Generation Functions
# ============================================================================

def create_ring_reservoir(N, spectral_radius=0.9):
    """Create simple ring reservoir"""
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i+1) % N] = 1.0
        if i > 0:
            W[i, i-1] = 0.5
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    W = W * (spectral_radius / current_radius)
    return W

def create_random_reservoir(N, density=0.1, spectral_radius=0.9):
    """Create random sparse reservoir"""
    W = np.random.randn(N, N) * (np.random.rand(N, N) < density)
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)
    return W

def create_small_world_reservoir(N, k=4, p=0.1, spectral_radius=0.9):
    """Create small-world reservoir (simplified Watts-Strogatz)"""
    W = np.zeros((N, N))
    # Create ring lattice with k nearest neighbors
    for i in range(N):
        for j in range(1, k//2 + 1):
            W[i, (i+j) % N] = np.random.randn()
            W[i, (i-j) % N] = np.random.randn()
    # Rewire with probability p
    for i in range(N):
        for j in range(N):
            if W[i,j] != 0 and np.random.rand() < p:
                new_j = np.random.randint(N)
                W[i, new_j] = W[i, j]
                W[i, j] = 0
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)
    return W

def create_scale_free_reservoir(N, m=2, spectral_radius=0.9):
    """Create scale-free reservoir (simplified Barabási-Albert)"""
    W = np.zeros((N, N))
    # Start with small connected graph
    for i in range(m):
        for j in range(i):
            W[i,j] = np.random.randn()
            W[j,i] = np.random.randn()
    
    # Add nodes with preferential attachment
    for i in range(m, N):
        degrees = np.sum(np.abs(W) > 0, axis=1) + 1
        probs = degrees / np.sum(degrees)
        targets = np.random.choice(i, size=min(m, i), replace=False, p=probs[:i]/probs[:i].sum())
        for j in targets:
            W[i,j] = np.random.randn()
            W[j,i] = np.random.randn()
    
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)
    return W

# ============================================================================
# Memory Capacity Measurement
# ============================================================================

def measure_memory_capacity(W, W_in, max_delay=30, train_len=1500, test_len=500):
    """Measure memory capacity"""
    N = W.shape[0]
    
    u_train = np.random.uniform(-1, 1, train_len)
    u_test = np.random.uniform(-1, 1, test_len)
    
    # Collect states
    x = np.zeros(N)
    X_train = np.zeros((train_len, N))
    for t in range(train_len):
        x = np.tanh(W @ x + W_in.flatten() * u_train[t])
        X_train[t] = x
    
    x = np.zeros(N)
    X_test = np.zeros((test_len, N))
    for t in range(test_len):
        x = np.tanh(W @ x + W_in.flatten() * u_test[t])
        X_test[t] = x
    
    # Measure capacity
    MC = np.zeros(max_delay)
    for k in range(max_delay):
        if k >= train_len - 10 or k >= test_len - 10:
            break
        
        y_train = u_train[k:train_len]
        y_test = u_test[k:test_len]
        X_train_k = X_train[:len(y_train)]
        X_test_k = X_test[:len(y_test)]
        
        # Ridge regression (manual implementation)
        alpha = 1e-6
        W_out = np.linalg.solve(X_train_k.T @ X_train_k + alpha * np.eye(N), 
                                X_train_k.T @ y_train)
        y_pred = X_test_k @ W_out
        
        # Squared correlation
        cov = np.cov(y_test, y_pred)[0, 1]
        var_y = np.var(y_test)
        var_pred = np.var(y_pred)
        
        if var_y > 0 and var_pred > 0:
            MC[k] = (cov ** 2) / (var_y * var_pred)
    
    return MC

def analyze_spectrum(W):
    """Analyze spectral properties"""
    eigenvalues = np.linalg.eigvals(W)
    sorted_mags = np.sort(np.abs(eigenvalues))[::-1]
    
    return {
        'eigenvalues': eigenvalues,
        'spectral_radius': np.max(np.abs(eigenvalues)),
        'spectral_gap': sorted_mags[0] - sorted_mags[1] if len(sorted_mags) > 1 else 0,
        'effective_rank': (np.sum(sorted_mags) ** 2) / np.sum(sorted_mags ** 2)
    }

# ============================================================================
# Run Experiments
# ============================================================================

N = 100
spectral_radii = [0.5, 0.7, 0.9, 0.95, 0.99]
topologies = ['ring', 'random', 'small_world', 'scale_free']

results = {top: {sr: None for sr in spectral_radii} for top in topologies}
spectral_properties = {top: {sr: None for sr in spectral_radii} for top in topologies}

print(f"\nReservoir size: {N}")
print(f"Running experiments...\n")

for topology in topologies:
    print(f"Topology: {topology}")
    for sr in spectral_radii:
        if topology == 'ring':
            W = create_ring_reservoir(N, spectral_radius=sr)
        elif topology == 'random':
            W = create_random_reservoir(N, density=0.1, spectral_radius=sr)
        elif topology == 'small_world':
            W = create_small_world_reservoir(N, k=4, p=0.1, spectral_radius=sr)
        elif topology == 'scale_free':
            W = create_scale_free_reservoir(N, m=2, spectral_radius=sr)
        
        W_in = np.random.randn(N, 1) * 0.1
        MC = measure_memory_capacity(W, W_in, max_delay=30, train_len=1500, test_len=500)
        results[topology][sr] = MC
        spectral_properties[topology][sr] = analyze_spectrum(W)
        
        print(f"  ρ={sr:.2f}: Total MC={np.sum(MC):.3f}")

# ============================================================================
# Generate Figures
# ============================================================================

# Figure 1: Memory capacity curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
sr_plot = 0.9

for idx, topology in enumerate(topologies):
    MC = results[topology][sr_plot]
    axes[idx].plot(range(len(MC)), MC, 'o-', linewidth=2, markersize=4)
    axes[idx].set_xlabel('Delay k', fontsize=11)
    axes[idx].set_ylabel('Memory Capacity $MC_k$', fontsize=11)
    axes[idx].set_title(f'{topology.replace("_", " ").title()} (ρ={sr_plot})', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('outputs/open_research_20251002_180900/memory_capacity_by_topology.pdf', dpi=300, bbox_inches='tight')
print("\nSaved: memory_capacity_by_topology.pdf")

# Figure 2: Total MC vs spectral radius
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for topology in topologies:
    total_MCs = [np.sum(results[topology][sr]) for sr in spectral_radii]
    ax.plot(spectral_radii, total_MCs, 'o-', linewidth=2, markersize=8, 
            label=topology.replace('_', ' ').title())

ax.set_xlabel('Spectral Radius ρ', fontsize=13)
ax.set_ylabel('Total Memory Capacity', fontsize=13)
ax.set_title('Memory Capacity vs Spectral Radius', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/open_research_20251002_180900/total_mc_vs_spectral_radius.pdf', dpi=300, bbox_inches='tight')
print("Saved: total_mc_vs_spectral_radius.pdf")

# Figure 3: Eigenvalue distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, topology in enumerate(topologies):
    spec = spectral_properties[topology][sr_plot]
    eigs = spec['eigenvalues']
    axes[idx].scatter(np.real(eigs), np.imag(eigs), alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    theta = np.linspace(0, 2*np.pi, 100)
    axes[idx].plot(sr_plot * np.cos(theta), sr_plot * np.sin(theta), 'r--', linewidth=2)
    axes[idx].set_xlabel('Real part', fontsize=11)
    axes[idx].set_ylabel('Imaginary part', fontsize=11)
    axes[idx].set_title(f'{topology.replace("_", " ").title()}', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/open_research_20251002_180900/eigenvalue_distributions.pdf', dpi=300, bbox_inches='tight')
print("Saved: eigenvalue_distributions.pdf")

# Figure 4: Spectral gap vs MC
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for topology in topologies:
    gaps = [spectral_properties[topology][sr]['spectral_gap'] for sr in spectral_radii]
    mcs = [np.sum(results[topology][sr]) for sr in spectral_radii]
    ax.plot(gaps, mcs, 'o-', linewidth=2, markersize=8, label=topology.replace('_', ' ').title())

ax.set_xlabel('Spectral Gap (λ₁ - λ₂)', fontsize=13)
ax.set_ylabel('Total Memory Capacity', fontsize=13)
ax.set_title('Memory Capacity vs Spectral Gap', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/open_research_20251002_180900/mc_vs_spectral_gap.pdf', dpi=300, bbox_inches='tight')
print("Saved: mc_vs_spectral_gap.pdf")

print("\n" + "="*60)
print("All experiments and visualizations completed!")
print("="*60)
