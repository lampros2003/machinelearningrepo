import numpy as np
from scipy.stats import norm


def compute_cdf_difference(y_points, x):
    """
    Compute H(y|x) - H(y-1|x) for the given data model Y = 0.8X + W
    where W is N(0,1)
    """
    # For Y = 0.8X + W, Y|X follows N(0.8X, 1)
    mean = 0.8 * x
    std = 1.0
   
    cdf_values = norm.cdf(y_points, loc=mean, scale=std)
    # Calculate differences for trapezoidal rule
    cdf_diff = np.diff(cdf_values)
    return cdf_diff


def numerical_conditional_expectation(x, g_func, y_min=-10, y_max=10, n_points=1000):
    """
    Compute E[G(Y)|X=x] using trapezoidal numerical integration
   
    Parameters:
    x: point at which to compute conditional expectation
    g_func: function G(Y) to apply
    y_min, y_max: integration limits
    n_points: number of points for numerical integration
    """
    # Create grid of y points
    y_points = np.linspace(y_min, y_max, n_points)
   
    # Compute G(Y) values
    g_values = g_func(y_points)
   
    # Compute CDF differences for integration
    cdf_diffs = compute_cdf_difference(y_points, x)
   
    # Apply trapezoidal rule
    # Average consecutive G(Y) values and multiply by CDF differences
    g_avg = (g_values[1:] + g_values[:-1]) / 2
    result = np.sum(g_avg * cdf_diffs)
   
    return result


# Define the two G(Y) functions from the problem
def g1(y):
    """G(Y) = Y"""
    return y


def g2(y):
    """G(Y) = min{1, max{-1, Y}}"""
    return np.clip(y, -1, 1)


# Function to compute expectations over a range of X values
def compute_expectations_over_range(x_range):
    """Compute both conditional expectations over a range of X values"""
    e1 = np.array([numerical_conditional_expectation(x, g1) for x in x_range])
    e2 = np.array([numerical_conditional_expectation(x, g2) for x in x_range])
    return e1, e2


# Generate results
def plot_numerical_results():
    x_range = np.linspace(-5, 5, 100)
    e1, e2 = compute_expectations_over_range(x_range)
   
    return x_range, e1, e2


if __name__ == "__main__":
    x_range, e1, e2 = plot_numerical_results()


if __name__ == "__main__":
    # Example usage
    x_range = np.linspace(-10, 10, 100)
    expectations1, expectations2 = compute_expectations_over_range(x_range)
   
    # Plotting
    import matplotlib.pyplot as plt
   
    plt.figure(figsize=(12, 5))
   
    plt.subplot(1, 2, 1)
    plt.plot(x_range, expectations1)
    plt.title('E[Y|X=x]')
    plt.xlabel('x')
    plt.ylabel('Conditional Expectation')
    plt.grid(True)
   
    plt.subplot(1, 2, 2)
    plt.plot(x_range, expectations2)
    plt.title('E[min{1, max{-1, Y}}|X=x]')
    plt.xlabel('x')
    plt.ylabel('Conditional Expectation')
    plt.grid(True)
   
    plt.tight_layout()
    plt.show()

