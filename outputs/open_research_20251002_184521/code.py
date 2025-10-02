import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

output_dir = "."

# Visualize the different topologies
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

n_nodes = 50  # Smaller for visualization

def create_topology_graph(pattern, n_nodes):
    """Create networkx graph for visualization"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    if pattern == 'random':
        density = 0.1
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and np.random.rand() < density:
                    G.add_edge(i, j)
    
    elif pattern == 'block_diagonal':
        n_blocks = 4
        block_size = n_nodes // n_blocks
        # Dense within blocks
        for block in range(n_blocks):
            start = block * block_size
            end = start + block_size
            for i in range(start, end):
                for j in range(start, end):
                    if i != j and np.random.rand() < 0.3:
                        G.add_edge(i, j)
        # Sparse between blocks
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i // block_size != j // block_size and np.random.rand() < 0.01:
                    G.add_edge(i, j)
    
    elif pattern == 'hierarchical':
        # Local connections
        for i in range(n_nodes):
            for j in range(max(0, i-2), min(n_nodes, i+3)):
                if i != j:
                    G.add_edge(i, j)
        # Long-range connections
        n_long_range = int(n_nodes * 5)
        for _ in range(n_long_range):
            i = np.random.randint(0, n_nodes)
            distance = int(np.random.power(2) * n_nodes / 2)
            j = (i + distance) % n_nodes
            G.add_edge(i, j)
    
    elif pattern == 'small_world':
        k = 4
        for i in range(n_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n_nodes
                G.add_edge(i, neighbor)
        # Rewire
        p = 0.3
        edges = list(G.edges())
        for i, j in edges:
            if np.random.rand() < p:
                G.remove_edge(i, j)
                new_j = np.random.randint(0, n_nodes)
                if new_j != i and not G.has_edge(i, new_j):
                    G.add_edge(i, new_j)
    
    return G

patterns = ['random', 'block_diagonal', 'hierarchical', 'small_world']
titles = ['Random Sparse', 'Block-Diagonal', 'Hierarchical', 'Small-World']

np.random.seed(42)

for idx, (pattern, title) in enumerate(zip(patterns, titles)):
    ax = axes[idx]
    G = create_topology_graph(pattern, n_nodes)
    
    if pattern == 'block_diagonal':
        # Arrange nodes in blocks
        pos = {}
        block_size = n_nodes // 4
        for block in range(4):
            block_nodes = range(block * block_size, (block + 1) * block_size)
            # Circular layout for each block
            for i, node in enumerate(block_nodes):
                angle = 2 * np.pi * i / block_size
                radius = 0.3
                center_x = np.cos(2 * np.pi * block / 4) * 0.6
                center_y = np.sin(2 * np.pi * block / 4) * 0.6
                pos[node] = (center_x + radius * np.cos(angle), 
                            center_y + radius * np.sin(angle))
    elif pattern == 'small_world':
        # Circular layout
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue', 
                          ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, arrows=False,
                          edge_color='gray', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "topology_structures.png"), dpi=300, bbox_inches='tight')
print("Saved topology_structures.png")
plt.close()

# Create a supplementary figure showing spectral properties
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Generate reservoirs and analyze spectral properties
reservoir_size = 200
patterns = ['random', 'block_diagonal', 'hierarchical', 'small_world']
colors = ['blue', 'red', 'green', 'orange']

from scipy.linalg import eig as scipy_eig

spectral_data = {p: [] for p in patterns}

for pattern in patterns:
    for trial in range(5):
        if pattern == 'random':
            from __main__ import ReservoirComputer
            rc = ReservoirComputer(input_size=1, reservoir_size=reservoir_size, 
                                 output_size=1, connectivity_pattern=pattern)
        else:
            rc = ReservoirComputer(input_size=1, reservoir_size=reservoir_size, 
                                 output_size=1, connectivity_pattern=pattern)
        
        eigenvalues = np.linalg.eigvals(rc.W_res)
        spectral_data[pattern].append(eigenvalues)

# Plot eigenvalue distributions
ax = axes[0]
for pattern, color in zip(patterns, colors):
    all_eigs = np.concatenate(spectral_data[pattern])
    real_parts = np.real(all_eigs)
    imag_parts = np.imag(all_eigs)
    ax.scatter(real_parts, imag_parts, alpha=0.3, s=1, label=pattern, color=color)

circle = plt.Circle((0, 0), 0.9, fill=False, linestyle='--', color='black', linewidth=2)
ax.add_patch(circle)
ax.set_xlabel('Real part', fontsize=12)
ax.set_ylabel('Imaginary part', fontsize=12)
ax.set_title('Eigenvalue Distribution (spectral radius = 0.9)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# Plot effective rank over time
ax = axes[1]
for pattern, color in zip(patterns, colors):
    rc = ReservoirComputer(input_size=1, reservoir_size=reservoir_size, 
                         output_size=1, connectivity_pattern=pattern)
    
    # Run reservoir with random input and measure rank
    n_steps = 500
    u = np.random.rand(n_steps, 1) * 2 - 1
    states = rc.run_reservoir(u)
    
    # Compute effective rank using singular values
    ranks = []
    for t in range(50, n_steps, 10):
        state_matrix = states[t-50:t, :]
        singular_values = np.linalg.svd(state_matrix, compute_uv=False)
        # Effective rank: sum of normalized singular values
        sv_normalized = singular_values / np.sum(singular_values)
        entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-10))
        eff_rank = np.exp(entropy)
        ranks.append(eff_rank)
    
    time_points = range(50, n_steps, 10)
    ax.plot(time_points, ranks, label=pattern, color=color, linewidth=2)

ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Effective Rank', fontsize=12)
ax.set_title('State Space Dimensionality Over Time', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectral_analysis.png"), dpi=300, bbox_inches='tight')
print("Saved spectral_analysis.png")
plt.close()

print("\nAll figures generated successfully!")
