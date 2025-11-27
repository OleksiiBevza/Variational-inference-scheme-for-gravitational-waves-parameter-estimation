import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner


"""
Sourse: https://github.com/ThibeauWouters?tab=repositories
"""

np.random.seed(0)
"""The following can be used to improve plotting a bit"""
params = {
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16
}

plt.rcParams.update(params)

# Improved corner kwargs -- pass them to corner.corner
default_corner_kwargs = dict(bins=50, 
                        smooth=0.6, 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

"""Now for the main function"""

def generate_gaussian_mixture(n_dim: int,
                              n_gaussians: int = 2,
                              n_samples: int = 10_000,
                              means: list = None,
                              covariances: list = None,
                              weights: list = None,
                              width_mean: float = 10.0,
                              width_cov: float = 1.0):
    """
    Generate samples from a mixture of Gaussians. 
    This function generates samples from a Gaussian mixture model with specified means, covariances, and weights.
    If means, covariances, or weights are not provided, they are generated randomly.
    
    Args:
        n_dim (int): The number of dimensions for the samples.
        n_gaussians (int, optional): The number of Gaussian components in the mixture. Defaults to 1.
        n_samples (int, optional): The number of samples to generate. Defaults to 10,000.
        means (list, optional): The mean vectors. If not specified, then they will be generated randomly. Defaults to None.
        covariances (list, optional): The square covariance matrix of size (n_dim x n_dim). If not specified, then they will be generated randomly. Defaults to None.
        weights (list, optional): Weights between the different Gaussians. If not specified, equal weights are used. Defaults to None.
        width_mean (float, optional): The width of the mean distribution. Defaults to 10.0.
        width_cov (float, optional): The width of the covariance distribution. Defaults to 1.0.
    """
    
    # If no mean vector is given, generate random means
    seed = np.random.randint(0, 1000)
    jax_key = jax.random.PRNGKey(seed)
    if means is None:
        means = []
        for _ in range(n_gaussians):
            # Split the key to ensure different means for each Gaussian
            jax_key, subkey = jax.random.split(jax_key)
            this_means = jax.random.uniform(subkey, (n_dim,), minval=-width_mean, maxval=width_mean)
            print("this_means")
            print(this_means)
            
            means.append(this_means)
    print(f"Means: {means}")
        
    # If no covariance matrix is given, generate identity matrices
    if covariances is None:
        covariances = []
        for _ in range(n_gaussians):
            jax_key, subkey = jax.random.split(jax_key)
            A = jax.random.uniform(subkey, (n_dim, n_dim), minval=-width_cov, maxval=width_cov)
            B = jnp.dot(A, A.transpose())
            covariances.append(B)
    print(f"Covariances: {covariances}")
    
    # If no weights are given, use equal weights between the Gaussians
    if weights is None:
        weights = [1.0 / n_gaussians] * n_gaussians
    print(f"Weights: {weights}")
        
    # Check if everythingq is consistent
    if len(means) != n_gaussians or len(covariances) != n_gaussians or len(weights) != n_gaussians:
        raise ValueError("Means, covariances, and weights must match the number of Gaussians.")
    
    # Generate samples
    samples = []
    for i in range(n_samples):
        # Choose a Gaussian component based on weights
        this_key = jax.random.PRNGKey(i)
        this_key, sample_key = jax.random.split(this_key)
        component = np.random.choice(n_gaussians, p=weights)
        mean = means[component]
        covariance = covariances[component]
        
        # Generate a sample from the chosen Gaussian
        sample = jax.random.multivariate_normal(sample_key, mean, covariance)
        samples.append(sample)
        
    samples = jnp.array(samples)
    return samples

def main():
    """Small main function to test the Gaussian mixture generation."""
    
    n_dim = 15
    n_gaussians = 2
    weights = None
    # weights = [0.35, 0.15, 0.50]
    # weights = [0.2, 0.2, 0.15, 0.25, 0.2]
    

    samples = generate_gaussian_mixture(n_dim, n_gaussians, weights=weights)

    # Plot the samples
    corner.corner(np.array(samples), labels=[f'x{i+1}' for i in range(n_dim)], **default_corner_kwargs)
    #if weights is None:
        #save_name = f'./figures/gmm_ndim_{n_dim}_ngauss_{n_gaussians}.png'
    #else:
        #save_name = f'./figures/gmm_ndim_{n_dim}_ngauss_{n_gaussians}_weights.png'
    #plt.savefig(save_name)
    #plt.close()
    
if __name__ == "__main__":
    main()