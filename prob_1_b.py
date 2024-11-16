import numpy as np
from scipy.stats import norm

def f0(x):
    """Probability density function for H0"""
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

def f1(x):
    """Probability density function for H1"""
    return 0.5 * (norm.pdf(x, -1, 1) + norm.pdf(x, 1, 1))

def bayes_test(x1, x2,simplified ="simple"):
    """Optimal Bayes test"""
    if simplified == "simple":
        log_likelihood_ratio = np.log(np.cosh(x1)*np.cosh(x2))
        if log_likelihood_ratio > np.log(np.e):
            return 1  # Decide H1
        else:
            return 0  # Decide H0
    
    log_likelihood_ratio = np.log(f1(x1) / f0(x1)) + np.log(f1(x2) / f0(x2))
    if log_likelihood_ratio > 0:
        return 1  # Decide H1
    else:
        return 0  # Decide H0

num_samples = 10**6

# Generate samples from H0 and H1
x0_1, x0_2 = np.random.normal(0, 1, (2, num_samples))
x1_1 = np.random.choice([-1, 1], num_samples, p=[0.5, 0.5]) + np.random.normal(0, 1, num_samples)
x1_2 = np.random.choice([-1, 1], num_samples, p=[0.5, 0.5]) + np.random.normal(0, 1, num_samples)

# Apply the Bayes test
y0 = [bayes_test(x0_1[i], x0_2[i]) for i in range(num_samples)]
y1 = [bayes_test(x1_1[i], x1_2[i]) for i in range(num_samples)]

# Calculate error probabilities
error_p0 = np.sum(y0) / num_samples
error_p1 = 1-np.sum(y1) / num_samples
total_error = 0.5 * error_p0 + 0.5 * error_p1

print(f"Error probability for H0: {error_p0:.4f}")
print(f"Error probability for H1: {error_p1:.4f}")
print(f"Total error probability: {total_error:.4f}")