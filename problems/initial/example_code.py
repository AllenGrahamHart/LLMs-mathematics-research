# Example custom initial code
import numpy as np
import matplotlib.pyplot as plt

# This code will be modified during research iterations

def example_function():
    """Example function that can be built upon."""
    print("Starting with custom initial code")

    # Example usage of imports (ready for expansion)
    data = np.array([1, 2, 3, 4, 5])
    print(f"Example data shape: {data.shape}")

    return True

def example_plot():
    """Example plotting function."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Example Plot')
    plt.grid(True)
    # plt.savefig('artifacts/figures/example.png', dpi=300)

if __name__ == "__main__":
    example_function()
