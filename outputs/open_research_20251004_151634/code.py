import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

np.random.seed(42)
output_dir = "."

print("="*70)
print("Final Experiments: Optimal Regularization in Echo State Networks")
print("="*70)

# ============================================
# Generate Lorenz System Data
# ============================================
print("\n1. Generating Lorenz system data...")

def lorenz(state, t, sigma=10.0, beta=8/3, rho=28.0):
    xi, upsilon, zeta = state
    dxi = sigma * (upsilon - xi)
    dupsilon = xi * (rho - zeta) - upsilon
    dzeta = xi * upsilon - beta * zeta
    return [dxi, dupsilon, dzeta]

dt = 0.01
t_long = np.arange(0, 500, dt)  # 50000 points
initial_state = [0, 1.0, 1.05]
trajectory = odeint(lorenz, initial_state, t_long)

xi_data = trajectory[:, 0]
zeta_data = trajectory[:, 2]
print(f"   Generated {len(xi_data)} data points")

# ============================================
# Echo State Network Setup
# ============================================
print("\n2. Setting up Echo State Network...")

def create_esn(N_reservoir=300, spectral_radius=1.0, input_scaling=0.05):
    A = np.random.randn(N_reservoir, N_reservoir)
    eigenvalues = np.linalg.eigvals(A)
    A = A * (spectral_radius / np.max(np.abs(eigenvalues)))
    C = np.random.uniform(-input_scaling, input_scaling, (N_reservoir, 1))
    b = np.random.uniform(-input_scaling, input_scaling, N_reservoir)
    return A, C, b

def run_esn(A, C, b, inputs, x0=None):
    N = A.shape[0]
    n_steps = len(inputs)
    X = np.zeros((n_steps, N))
    x = np.zeros(N) if x0 is None else x0.copy()
    
    for i in range(n_steps):
        z = inputs[i]
        x = np.tanh(A @ x + C.flatten() * z + b)
        X[i] = x
    return X

def train_esn_readout(X, targets, lambda_reg):
    XTX = X.T @ X
    XTy = X.T @ targets
    W = np.linalg.solve(XTX + lambda_reg * np.eye(X.shape[1]), XTy)
    return W

A, C, b = create_esn(N_reservoir=300, spectral_radius=1.0, input_scaling=0.05)
print(f"   ESN: {A.shape[0]} neurons")

X_all = run_esn(A, C, b, xi_data)
print("   Reservoir states computed")

# Test set
TEST_START = 20000
TEST_END = 40000
X_test = X_all[TEST_START:TEST_END]
y_test = zeta_data[TEST_START:TEST_END]
print(f"   Test set: {len(y_test)} points")

# ============================================
# Experiment 1: Vary λ for Fixed ℓ
# ============================================
print("\n3. Experiment 1: Varying λ for fixed ℓ...")

ell_values = [1000, 3000, 6000, 10000]
lambda_values = np.logspace(-16, -3, 50)  # EXTENDED RANGE

results_exp1 = {}

for ell in ell_values:
    print(f"   ℓ = {ell}...", end=" ")
    X_train = X_all[:ell]
    y_train = zeta_data[:ell]
    
    errors = []
    for lam in lambda_values:
        W = train_esn_readout(X_train, y_train, lam)
        predictions_test = X_test @ W
        rmse = np.sqrt(np.mean((predictions_test - y_test)**2))
        errors.append(rmse)
    
    results_exp1[ell] = np.array(errors)
    optimal_idx = np.argmin(errors)
    print(f"min RMSE = {np.min(errors):.4f} at λ = {lambda_values[optimal_idx]:.2e}")

plt.figure(figsize=(10, 6))
for ell in ell_values:
    plt.loglog(lambda_values, results_exp1[ell], 'o-', label=f'ℓ = {ell}', 
               alpha=0.7, markersize=4, linewidth=1.5)

plt.xlabel('Regularization parameter λ', fontsize=13)
plt.ylabel('Test RMSE', fontsize=13)
plt.title('Test Error vs. Regularization for Different Training Lengths', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiment1_lambda_vs_error.png", dpi=300, bbox_inches='tight')
print("   → Figure saved: experiment1_lambda_vs_error.png")

# ============================================
# Experiment 2: Optimal λ Scaling
# ============================================
print("\n4. Experiment 2: Analyzing optimal λ scaling...")

ell_range = np.arange(500, 15000, 500)
optimal_lambdas = []
lambda_search = np.logspace(-16, -3, 50)

for ell in ell_range:
    X_train = X_all[:ell]
    y_train = zeta_data[:ell]
    
    errors = []
    for lam in lambda_search:
        W = train_esn_readout(X_train, y_train, lam)
        predictions_test = X_test @ W
        rmse = np.sqrt(np.mean((predictions_test - y_test)**2))
        errors.append(rmse)
    
    optimal_lambda = lambda_search[np.argmin(errors)]
    optimal_lambdas.append(optimal_lambda)

optimal_lambdas = np.array(optimal_lambdas)

# Fit power law
log_ell = np.log(ell_range)
log_lambda = np.log(optimal_lambdas)
coeffs = np.polyfit(log_ell, log_lambda, 1)
alpha_fitted = -coeffs[0]
C_fitted = np.exp(coeffs[1])

print(f"   Fitted: λ* = {C_fitted:.2e} × ℓ^(-{alpha_fitted:.3f})")
print(f"   Theory: λ* ~ ℓ^(-1/3) = ℓ^(-0.333)")
print(f"   Discrepancy: Δα = {abs(alpha_fitted - 1/3):.3f}")

plt.figure(figsize=(10, 6))
plt.loglog(ell_range, optimal_lambdas, 'o', label='Empirical optimal λ*', 
           markersize=7, alpha=0.7, color='C0')
plt.loglog(ell_range, C_fitted * ell_range**(-alpha_fitted), '--', 
           label=f'Fitted: {C_fitted:.2e} ℓ^(-{alpha_fitted:.2f})', linewidth=2.5, color='C1')
plt.loglog(ell_range, 0.05 * ell_range**(-1/3), ':', 
           label='Theory: C ℓ^(-1/3)', linewidth=2.5, color='C2')

plt.xlabel('Training length ℓ', fontsize=13)
plt.ylabel('Optimal λ*', fontsize=13)
plt.title('Optimal Regularization Parameter vs. Training Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiment2_optimal_scaling.png", dpi=300, bbox_inches='tight')
print("   → Figure saved: experiment2_optimal_scaling.png")

# ============================================
# Experiment 3: Adaptive vs Fixed
# ============================================
print("\n5. Experiment 3: Adaptive vs. fixed regularization...")

ell_range_comp = np.arange(1000, 15000, 500)
results_adaptive = []
results_fixed = []

for ell in ell_range_comp:
    X_train = X_all[:ell]
    y_train = zeta_data[:ell]
    
    # Adaptive
    lambda_adaptive = C_fitted * ell**(-alpha_fitted)
    W_adaptive = train_esn_readout(X_train, y_train, lambda_adaptive)
    pred_adaptive = X_test @ W_adaptive
    rmse_adaptive = np.sqrt(np.mean((pred_adaptive - y_test)**2))
    results_adaptive.append(rmse_adaptive)
    
    # Fixed
    lambda_fixed = 1e-8
    W_fixed = train_esn_readout(X_train, y_train, lambda_fixed)
    pred_fixed = X_test @ W_fixed
    rmse_fixed = np.sqrt(np.mean((pred_fixed - y_test)**2))
    results_fixed.append(rmse_fixed)

plt.figure(figsize=(10, 6))
plt.loglog(ell_range_comp, results_adaptive, 'o-', 
           label=f'Adaptive: λ = {C_fitted:.1e} ℓ^(-{alpha_fitted:.2f})', 
           markersize=6, linewidth=2.5, color='C0')
plt.loglog(ell_range_comp, results_fixed, 's-', 
           label='Fixed: λ = 10^(-8)', markersize=6, linewidth=2.5, color='C1')

# Reference
ell_ref = np.array([1000, 15000])
plt.loglog(ell_ref, 8.0 * ell_ref**(-2/3), '--', color='gray', 
           label='Reference: ℓ^(-2/3)', linewidth=2, alpha=0.6)

plt.xlabel('Training length ℓ', fontsize=13)
plt.ylabel('Test RMSE', fontsize=13)
plt.title('Adaptive vs. Fixed Regularization Strategy', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiment3_adaptive_vs_fixed.png", dpi=300, bbox_inches='tight')
print("   → Figure saved: experiment3_adaptive_vs_fixed.png")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Optimal λ scaling:     λ* = {C_fitted:.2e} × ℓ^(-{alpha_fitted:.3f})")
print(f"Theoretical prediction: λ* ~ ℓ^(-0.333)")
print(f"\nAt ℓ = {ell_range_comp[-1]}:")
print(f"  Adaptive RMSE: {results_adaptive[-1]:.6f}")
print(f"  Fixed RMSE:    {results_fixed[-1]:.6f}")
improvement = (results_fixed[-1] - results_adaptive[-1]) / results_fixed[-1] * 100
print(f"  Improvement:   {improvement:.1f}%")
print("="*70)
