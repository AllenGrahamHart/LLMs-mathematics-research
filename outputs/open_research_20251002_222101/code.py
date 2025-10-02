import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.linear_model import Ridge
import os

output_dir = "."
np.random.seed(42)

class TimeDelayReservoir:
    """Time-delay feedback reservoir computing architecture."""
    def __init__(self, input_dim, reservoir_dim, delay_length=0, 
                 spectral_radius=0.9, input_scaling=1.0, feedback_scaling=0.0):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.delay_length = delay_length
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        
        # Input weights
        self.W_in = np.random.randn(reservoir_dim, input_dim) * input_scaling
        
        # Reservoir weights with proper spectral radius
        self.W_res = np.random.randn(reservoir_dim, reservoir_dim)
        self.W_res *= 0.5
        eigenvalues = np.linalg.eigvals(self.W_res)
        self.W_res *= spectral_radius / np.max(np.abs(eigenvalues))
        
        # Delay feedback weights - only if delay > 0
        if delay_length > 0 and feedback_scaling > 0:
            self.W_delay = np.random.randn(reservoir_dim, reservoir_dim) * feedback_scaling
            eig_delay = np.linalg.eigvals(self.W_delay)
            self.W_delay *= 0.5 / max(np.abs(eig_delay))
        else:
            self.W_delay = None
            
    def initialize_state(self):
        self.state = np.zeros(self.reservoir_dim)
        if self.delay_length > 0:
            self.delay_buffer = [np.zeros(self.reservoir_dim) 
                                for _ in range(self.delay_length)]
        
    def step(self, input_signal):
        pre_activation = (np.dot(self.W_res, self.state) + 
                         np.dot(self.W_in, input_signal))
        
        if self.delay_length > 0 and self.W_delay is not None:
            delayed_state = self.delay_buffer[0]
            delay_contribution = np.dot(self.W_delay, delayed_state)
            pre_activation += delay_contribution
            self.delay_buffer.pop(0)
            self.delay_buffer.append(self.state.copy())
        
        self.state = np.tanh(pre_activation)
        return self.state
    
    def run(self, input_sequence, washout=100):
        self.initialize_state()
        states = []
        for t, inp in enumerate(input_sequence):
            state = self.step(inp)
            if t >= washout:
                states.append(state.copy())
        return np.array(states)

def compute_memory_capacity(reservoir, max_delay=50, test_length=5000):
    """Compute Jaeger's linear memory capacity."""
    input_signal = np.random.uniform(-0.5, 0.5, (test_length, 1))
    states = reservoir.run(input_signal, washout=200)
    
    memory_capacities = []
    for k in range(1, max_delay + 1):
        if len(states) < k + 10:
            break
        target = input_signal[200:200+len(states)-k]
        reservoir_states = states[k:]
        if len(target) < 100:
            break
        ridge = Ridge(alpha=1e-6)
        ridge.fit(reservoir_states, target)
        predictions = ridge.predict(reservoir_states)
        correlation = np.corrcoef(target.flatten(), predictions.flatten())[0, 1]
        mc_k = max(0, correlation ** 2)
        memory_capacities.append(mc_k)
    
    return np.array(memory_capacities)

def analyze_embedding_dimension(reservoir, input_sequence):
    """Compute effective embedding dimension via participation ratio."""
    states = reservoir.run(input_sequence, washout=200)
    U, s, Vh = np.linalg.svd(states.T, full_matrices=False)
    singular_values = s / np.sum(s)
    effective_dim = 1.0 / np.sum(singular_values ** 2)
    return singular_values, effective_dim

print("=== Time-Delay Feedback Reservoir Computing ===\n")

# Experimental parameters
delay_lengths = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
feedback_scalings = [0.1, 0.2, 0.3, 0.4, 0.5]
n_trials = 5
reservoir_dim = 200
spectral_radius = 0.9

print("[1/3] Memory capacity vs delay length...")
mc_means, mc_stds = [], []

for delay in delay_lengths:
    trial_mcs = []
    for trial in range(n_trials):
        np.random.seed(100 + trial * 10 + delay)
        fb_scale = 0.3 if delay > 0 else 0.0
        reservoir = TimeDelayReservoir(1, reservoir_dim, delay, spectral_radius, 1.0, fb_scale)
        mc = compute_memory_capacity(reservoir, max_delay=50, test_length=4000)
        trial_mcs.append(np.sum(mc))
    mc_means.append(np.mean(trial_mcs))
    mc_stds.append(np.std(trial_mcs))
    print(f"  τ={delay:2d}: MC = {mc_means[-1]:5.2f} ± {mc_stds[-1]:4.2f}")

print("\n[2/3] Feedback scaling interaction...")
scaling_results = np.zeros((len(delay_lengths), len(feedback_scalings)))
for i, delay in enumerate(delay_lengths):
    if delay == 0:
        continue
    for j, fb_scale in enumerate(feedback_scalings):
        np.random.seed(42 + i + j)
        reservoir = TimeDelayReservoir(1, reservoir_dim, delay, spectral_radius, 1.0, fb_scale)
        mc = compute_memory_capacity(reservoir, max_delay=40, test_length=3000)
        scaling_results[i, j] = np.sum(mc)

print("[3/3] Embedding dimension analysis...")
dim_means, dim_stds = [], []
for delay in delay_lengths:
    trial_dims = []
    for trial in range(n_trials):
        np.random.seed(200 + trial * 10 + delay)
        fb_scale = 0.3 if delay > 0 else 0.0
        test_signal = np.random.uniform(-0.5, 0.5, (3000, 1))
        reservoir = TimeDelayReservoir(1, reservoir_dim, delay, spectral_radius, 1.0, fb_scale)
        _, eff_dim = analyze_embedding_dimension(reservoir, test_signal)
        trial_dims.append(eff_dim)
    dim_means.append(np.mean(trial_dims))
    dim_stds.append(np.std(trial_dims))

# Create comprehensive 6-panel figure
fig = plt.figure(figsize=(15, 10))

ax1 = plt.subplot(2, 3, 1)
ax1.errorbar(delay_lengths, mc_means, yerr=mc_stds, fmt='o-', linewidth=2.5, 
             markersize=8, capsize=5, color='darkblue')
ax1.set_xlabel('Delay Length τ', fontsize=12)
ax1.set_ylabel('Total Memory Capacity', fontsize=12)
ax1.set_title('(a) Memory Capacity vs Delay', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=mc_means[0], color='red', linestyle='--', linewidth=2, alpha=0.6, label='Baseline (τ=0)')
ax1.legend(fontsize=10)

ax2 = plt.subplot(2, 3, 2)
im = ax2.imshow(scaling_results[1:], aspect='auto', cmap='viridis', origin='lower')
ax2.set_xlabel('Feedback Scaling α', fontsize=12)
ax2.set_ylabel('Delay Length τ', fontsize=12)
ax2.set_title('(b) MC vs Feedback Scaling', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(feedback_scalings)))
ax2.set_xticklabels([f'{s:.1f}' for s in feedback_scalings])
ax2.set_yticks(range(len(delay_lengths)-1))
ax2.set_yticklabels([str(d) for d in delay_lengths[1:]])
plt.colorbar(im, ax=ax2, label='Total MC')

ax3 = plt.subplot(2, 3, 3)
ax3.errorbar(delay_lengths, dim_means, yerr=dim_stds, fmt='s-', linewidth=2.5,
             markersize=8, capsize=5, color='darkred')
ax3.set_xlabel('Delay Length τ', fontsize=12)
ax3.set_ylabel('Effective Dimension $D_{\\mathrm{eff}}$', fontsize=12)
ax3.set_title('(c) Embedding Dimension vs Delay', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(dim_means, mc_means, s=150, c=delay_lengths, cmap='coolwarm', 
                      edgecolors='black', linewidth=1.5, alpha=0.8)
for i in [0, 2, 5, 9]:
    ax4.annotate(f'τ={delay_lengths[i]}', (dim_means[i], mc_means[i]), 
                fontsize=9, xytext=(6, 6), textcoords='offset points')
ax4.set_xlabel('Effective Dimension $D_{\\mathrm{eff}}$', fontsize=12)
ax4.set_ylabel('Total Memory Capacity', fontsize=12)
ax4.set_title('(d) Memory-Dimension Trade-off', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Delay τ')

ax5 = plt.subplot(2, 3, 5)
for delay in [0, 2, 5, 10]:
    np.random.seed(42 + delay)
    fb = 0.3 if delay > 0 else 0.0
    reservoir = TimeDelayReservoir(1, reservoir_dim, delay, spectral_radius, 1.0, fb)
    mc_curve = compute_memory_capacity(reservoir, max_delay=50, test_length=4000)
    ax5.plot(range(1, len(mc_curve)+1), mc_curve, label=f'τ={delay}', linewidth=2, alpha=0.8)
ax5.set_xlabel('Memory Lag k', fontsize=12)
ax5.set_ylabel('MC(k)', fontsize=12)
ax5.set_title('(e) Individual Memory Curves', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 50])

ax6 = plt.subplot(2, 3, 6)
for delay in [0, 2, 5, 10]:
    np.random.seed(42 + delay)
    fb = 0.3 if delay > 0 else 0.0
    test_sig = np.random.uniform(-0.5, 0.5, (3000, 1))
    reservoir = TimeDelayReservoir(1, reservoir_dim, delay, spectral_radius, 1.0, fb)
    sv, _ = analyze_embedding_dimension(reservoir, test_sig)
    ax6.semilogy(sv[:40], label=f'τ={delay}', linewidth=2, alpha=0.8)
ax6.set_xlabel('Component Index', fontsize=12)
ax6.set_ylabel('Normalized Singular Value', fontsize=12)
ax6.set_title('(f) Singular Value Spectra', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comprehensive_analysis.png", dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: comprehensive_analysis.png")

opt_idx = np.argmax(mc_means)
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Optimal delay:        τ* = {delay_lengths[opt_idx]}")
print(f"Maximum MC:           {mc_means[opt_idx]:.2f} ± {mc_stds[opt_idx]:.2f}")
print(f"Baseline MC (τ=0):    {mc_means[0]:.2f} ± {mc_stds[0]:.2f}")
improvement = ((mc_means[opt_idx] - mc_means[0]) / mc_means[0]) * 100
print(f"Improvement:          {improvement:.1f}%")
print(f"\nDimension range:      {min(dim_means):.1f} to {max(dim_means):.1f}")
print(f"Optimal dim (τ*):     {dim_means[opt_idx]:.1f}")
print(f"{'='*60}\n")
