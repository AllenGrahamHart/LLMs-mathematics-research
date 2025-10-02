import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import os

np.random.seed(42)
output_dir = "."

print("=" * 60)
print("ITERATION 5: FINAL FIXES & VALIDATION")
print("=" * 60)

class ReservoirComputer:
    """Echo State Network with flexible topology"""
    def __init__(self, input_dim, reservoir_dim, output_dim, 
                 spectral_radius=0.9, input_scaling=1.0, 
                 connectivity_type='random', sparsity=0.1):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        
        self.W_in = np.random.randn(reservoir_dim, input_dim) * input_scaling
        self.W = self._create_reservoir_matrix(connectivity_type, sparsity)
        self.W_out = None
        self.W_out_bias = None
        
    def _create_reservoir_matrix(self, conn_type, sparsity):
        N = self.reservoir_dim
        
        if conn_type == 'random':
            W = np.random.randn(N, N)
            mask = np.random.rand(N, N) > sparsity
            W[mask] = 0
        elif conn_type == 'ring':
            W = np.zeros((N, N))
            k = max(1, int(N * sparsity / 2))
            for i in range(N):
                for j in range(-k, k+1):
                    if j != 0:
                        W[i, (i+j) % N] = np.random.randn()
        elif conn_type == 'small_world':
            W = np.zeros((N, N))
            k = max(2, int(N * sparsity / 4))
            for i in range(N):
                for j in range(-k, k+1):
                    if j != 0:
                        W[i, (i+j) % N] = np.random.randn()
            n_shortcuts = int(N * sparsity)
            for _ in range(n_shortcuts):
                i, j = np.random.randint(0, N, 2)
                if i != j:
                    W[i, j] = np.random.randn()
        elif conn_type == 'hierarchical':
            W = np.zeros((N, N))
            block_size = max(5, N // 10)
            n_blocks = N // block_size
            for b in range(n_blocks):
                start = b * block_size
                end = min(start + block_size, N)
                block_mask = np.random.rand(end-start, end-start) < 0.3
                W[start:end, start:end] = np.random.randn(end-start, end-start) * block_mask
                if b < n_blocks - 1:
                    next_start = end
                    next_end = min(next_start + block_size, N)
                    inter_mask = np.random.rand(end-start, next_end-next_start) < 0.05
                    W[start:end, next_start:next_end] = np.random.randn(end-start, next_end-next_start) * inter_mask
        
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        return W
    
    def run(self, inputs, washout=100, return_all=False):
        T = len(inputs)
        states = np.zeros((T, self.reservoir_dim))
        x = np.zeros(self.reservoir_dim)
        
        for t in range(T):
            u = inputs[t].reshape(-1)
            x = np.tanh(self.W @ x + self.W_in @ u)
            states[t] = x
        
        if return_all:
            return states
        return states[washout:]
    
    def train(self, inputs, targets, washout=100, ridge_param=1e-6):
        states = self.run(inputs, washout=0, return_all=True)
        states = states[washout:]
        targets = targets[washout:]
        
        X = np.column_stack([states, np.ones(len(states))])
        XtX = X.T @ X
        XtX += ridge_param * np.eye(XtX.shape[0])
        Xty = X.T @ targets
        w = np.linalg.solve(XtX, Xty)
        
        self.W_out = w[:-1].T
        self.W_out_bias = w[-1]
        
        return states
    
    def predict(self, states):
        if self.W_out is None:
            raise ValueError("Model not trained yet")
        return states @ self.W_out.T + self.W_out_bias

def generate_narma(T, order=10):
    u = np.random.uniform(0, 0.5, T)
    y = np.zeros(T)
    
    for t in range(order, T):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-order:t]) + \
               1.5 * u[t-order] * u[t-1] + 0.1
    
    return u.reshape(-1, 1), y.reshape(-1, 1)

# COMPREHENSIVE STATISTICAL TEST WITH PROPER METHODOLOGY
print("\n1. COMPREHENSIVE NARMA-10 EVALUATION")
print("-" * 60)

topologies = ['random', 'ring', 'small_world', 'hierarchical']
colors = {'random': '#1f77b4', 'ring': '#ff7f0e', 'small_world': '#2ca02c', 'hierarchical': '#9467bd'}

# Multiple trials with DIFFERENT random seeds
all_results = {topo: [] for topo in topologies}

for topology in topologies:
    print(f"Evaluating {topology}...")
    for trial in range(20):  # More trials for better statistics
        # Different seed per trial
        np.random.seed(42 + trial)
        
        u_train, y_train = generate_narma(3000, order=10)
        u_test, y_test = generate_narma(1000, order=10)
        
        rc = ReservoirComputer(input_dim=1, reservoir_dim=200, output_dim=1,
                              connectivity_type=topology, sparsity=0.1, 
                              spectral_radius=0.9, input_scaling=1.0)
        
        rc.train(u_train, y_train, washout=100, ridge_param=1e-4)
        test_states = rc.run(u_test, washout=100)
        predictions = rc.predict(test_states)
        nrmse = np.sqrt(np.mean((predictions - y_test[100:])**2)) / np.std(y_test[100:])
        all_results[topology].append(nrmse)
    
    mean_nrmse = np.mean(all_results[topology])
    std_nrmse = np.std(all_results[topology])
    print(f"  NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")

# Statistical tests
print("\n2. STATISTICAL ANALYSIS")
print("-" * 60)

groups = [all_results[t] for t in topologies]
f_stat, p_value = f_oneway(*groups)
print(f"ANOVA: F={f_stat:.4f}, p={p_value:.6f}")

if p_value < 0.05:
    print("✓ Significant differences detected (p < 0.05)")
    result = tukey_hsd(*groups)
    print("\nPairwise comparisons (Tukey HSD):")
    for i, t1 in enumerate(topologies):
        for j, t2 in enumerate(topologies):
            if i < j:
                p_ij = result.pvalue[i, j]
                sig = "***" if p_ij < 0.001 else "**" if p_ij < 0.01 else "*" if p_ij < 0.05 else "ns"
                print(f"  {t1:15s} vs {t2:15s}: p={p_ij:.4f} {sig}")
else:
    print("✗ No significant differences detected (p >= 0.05)")
    print("  Performing effect size analysis...")
    # Calculate effect sizes
    for i, t1 in enumerate(topologies):
        for j, t2 in enumerate(topologies):
            if i < j:
                mean1, mean2 = np.mean(all_results[t1]), np.mean(all_results[t2])
                pooled_std = np.sqrt((np.std(all_results[t1])**2 + np.std(all_results[t2])**2) / 2)
                cohens_d = abs(mean1 - mean2) / pooled_std
                print(f"  {t1:15s} vs {t2:15s}: Cohen's d = {cohens_d:.3f}")

# 3. VISUALIZATION
print("\n3. CREATING FINAL VISUALIZATIONS")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Box plot comparison
ax = axes[0]
positions = np.arange(len(topologies))
bp = ax.boxplot([all_results[t] for t in topologies], positions=positions,
                labels=[t.replace('_', '-').title() for t in topologies],
                patch_artist=True, widths=0.6, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

for patch, topo in zip(bp['boxes'], topologies):
    patch.set_facecolor(colors[topo])
    patch.set_alpha(0.7)

ax.set_ylabel('NRMSE (lower is better)', fontsize=12, fontweight='bold')
ax.set_title('NARMA-10 Performance Distribution (N=200, 20 trials)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.25, 0.65])

# Add statistical annotation
if p_value < 0.05:
    ax.text(0.02, 0.98, f'ANOVA: p={p_value:.4f}*', transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.02, 0.98, f'ANOVA: p={p_value:.4f} (ns)', transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Right: Summary statistics table as visual
ax = axes[1]
ax.axis('off')

# Create summary table
table_data = []
table_data.append(['Topology', 'Mean NRMSE', 'Std Dev', 'Best Trial'])
for topo in topologies:
    mean_val = np.mean(all_results[topo])
    std_val = np.std(all_results[topo])
    best_val = np.min(all_results[topo])
    table_data.append([
        topo.replace('_', '-').title(),
        f'{mean_val:.4f}',
        f'{std_val:.4f}',
        f'{best_val:.4f}'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
for i, topo in enumerate(topologies, 1):
    table[(i, 0)].set_facecolor(colors[topo])
    table[(i, 0)].set_alpha(0.7)

ax.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("validation_and_guide.png", dpi=300, bbox_inches='tight')
print("✓ Saved: validation_and_guide.png")
plt.close()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFinal Statistical Result: p={p_value:.6f}")
if p_value < 0.05:
    print("✓ Significant topology effects confirmed")
else:
    print("✗ Topology effects not statistically significant at α=0.05")
    print("  Effect sizes suggest practical differences may exist")
