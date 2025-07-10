import numpy as np
import torch
from scipy.stats import multivariate_normal
import math

def evaluate_gaussian_ring_pdf(X, Y, k=3, sigma=0.5):
    """
    Compute PDF values on a meshgrid (X, Y) for a Gaussian ring.
    """
    grid_shape = X.shape
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape: (N, 2)

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    pdf_vals = np.zeros(coords.shape[0])
    for mu in centers:
        rv = multivariate_normal(mean=mu, cov=sigma**2 * np.eye(2))
        pdf_vals += rv.pdf(coords)

    return (pdf_vals / k).reshape(grid_shape)
    
def sample_gaussian_ring(n_samples=1000, k=3, sigma=0.5):
    """
    Samples from a Gaussian ring: a mixture of k Gaussians placed on the unit circle.
    Each Gaussian has isotropic covariance sigma^2 * I_2.
    """
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape: (k, 2)

    samples = []
    for _ in range(n_samples):
        idx = np.random.randint(0, k)
        center = centers[idx]
        point = np.random.multivariate_normal(center, sigma**2 * np.eye(2))
        samples.append(point)

    return np.array(samples, dtype=np.float32)


def sample_from_h_rejection(n_samples=1000, sigma=0.5):

    """
    Sample from h(x) ∝ exp(-[max(0, ||x|| - 1)]^2 / (2σ^2)) over R^2
    via rejection sampling from standard Gaussian as proposal.
    """
    samples = []
    while len(samples) < n_samples:
        x = np.random.normal(0, 1.5, size=(n_samples, 2))  # proposal
        norms = np.linalg.norm(x, axis=1)
        log_accept = -np.maximum(0, norms - 1)**2 / (2 * sigma**2)
        accept_prob = np.exp(log_accept - log_accept.max())  # normalize for numerical stability
        keep = np.random.rand(n_samples) < accept_prob
        accepted = x[keep]
        samples.extend(accepted.tolist())

    return np.array(samples[:n_samples], dtype=np.float32)


def sample_from_h_importance(n_samples=1000, sigma=0.5, proposal_std=1.5):
    """
    Importance sampling from h(x) ∝ exp(-[max(0, ||x|| - 1)]^2 / (2σ^2))
    using isotropic Gaussian proposal and resampling.
    """
    # Step 1: Draw samples from proposal (Gaussian)
    x = np.random.normal(loc=0, scale=proposal_std, size=(n_samples * 5, 2))  # oversample

    # Step 2: Compute log unnormalized importance weights
    norms = np.linalg.norm(x, axis=1)
    log_weights = -np.maximum(0, norms - 1)**2 / (2 * sigma**2)

    # Step 3: Convert to normalized importance weights
    weights = np.exp(log_weights - np.max(log_weights))  # numerical stability
    weights /= np.sum(weights)

    # Step 4: Resample according to weights
    indices = np.random.choice(len(x), size=n_samples, replace=True, p=weights)
    samples = x[indices]

    return samples.astype(np.float32)

# def evaluate_h_density(x, sigma=0.5):
#     """
#     Evaluates unnormalized h(x) ∝ exp(-[max(0, ||x|| - 1)]^2 / (2σ^2)) at tensor x
#     """
#     norm = torch.norm(x, dim=1)
#     penalty = torch.clamp(norm - 1, min=0)
#     c2 = 3.7724 
#     return (torch.exp(-penalty**2 / (2 * sigma**2)))/c2

def evaluate_h_density(x, sigma=0.5, c2=3.7724):
    """
    Returns the normalized density h(x) = (1 / (2πσ²)) * exp(...) / c2
    """
    norm = torch.norm(x, dim=1)
    penalty = torch.clamp(norm - 1, min=0)
    h_tilde = (1 / (2 * math.pi * sigma**2)) * torch.exp(-penalty**2 / (2 * sigma**2))
    return h_tilde / c2
