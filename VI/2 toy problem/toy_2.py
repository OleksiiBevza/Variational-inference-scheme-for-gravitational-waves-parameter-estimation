import jax
import jax.numpy as jnp
import equinox

from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Stack,
    Tanh,
)
from flowjax.distributions import StandardNormal, Transformed
from flowjax.flows import block_neural_autoregressive_flow
from paramax.wrappers import non_trainable

# Sourse: https://github.com/mdmould/gwax/blob/main/gwax/flows.py
class Flow:
    def __init__(self, bounds=None):
        self.bounds = bounds

    @staticmethod
    def Affine(loc=0, scale=1):
        affine = AffinePositiveScale(loc, scale)
        loc, scale = jnp.broadcast_arrays(
            affine.loc, jnp.asarray(scale, dtype=float),
        )
        affine = equinox.tree_at(lambda tree: tree.scale, affine, scale)
        return affine

    @staticmethod
    def Logistic(shape=()):
        loc = jnp.ones(shape) * 0.5
        scale = jnp.ones(shape) * 0.5
        return Chain([Tanh(shape), Flow.Affine(loc, scale)])

    @staticmethod
    def UnivariateBounder(bounds=None):
        # no bounds
        if (bounds is None) or all(bound is None for bound in bounds):
            return Identity()

        # bounded on one side
        elif any(bound is None for bound in bounds):
            # bounded on right-hand side
            if bounds[0] is None:
                loc = bounds[1]
                scale = -1
            # bounded on left-hand side
            elif bounds[1] is None:
                loc = bounds[0]
                scale = 1
            return Chain([Exp(), Flow.Affine(loc, scale)])

        # bounded on both sides
        else:
            loc = bounds[0]
            scale = bounds[1] - bounds[0]
            return Chain([Flow.Logistic(), Flow.Affine(loc, scale)])

    def Bounder(self):
        return Stack(list(map(self.UnivariateBounder, self.bounds)))

    def bound_from_unbound(self, flow):
        bounder = self.Bounder()

        if all(type(b) is Identity for b in bounder.bijections):
            bijection = flow.bijection
        else:
            bijection = Chain([flow.bijection, non_trainable(bounder)])

        return Transformed(non_trainable(flow.base_dist), bijection)

    def default_flow(self, key, **kwargs):
        default_kwargs = dict(
            key=key,
            base_dist=StandardNormal(shape=(len(self.bounds),)),
            invert=False,
            nn_depth=1,
            nn_block_dim=8,
            flow_layers=1,
        )

        for arg in kwargs:
            default_kwargs[arg] = kwargs[arg]

        flow = block_neural_autoregressive_flow(**default_kwargs)

        return self.bound_from_unbound(flow)




import sys
import time
import tqdm

import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax

from flowjax.distributions import Uniform
from paramax.wrappers import NonTrainable



# Sourse: https://github.com/mdmould/gwax/blob/main/gwax/flows.py
class VI:

    def __init__(self, prior_bounds, likelihood=None):
        self.prior_bounds = prior_bounds
        self.likelihood = likelihood

    def get_prior(self, bounds):
        lo = jnp.array(bounds)[:, 0]
        hi = jnp.array(bounds)[:, 1]
        return Uniform(minval=lo, maxval=hi)

    def get_log_likelihood(self, likelihood=None, return_variance=False):
        if likelihood is None:
            if return_variance:
                return lambda parameters: (0.0, 0.0)
            return lambda parameters: 0.0

        if return_variance:
            def log_likelihood_and_variance(parameters):
                likelihood.parameters.update(parameters)
                return likelihood.ln_likelihood_and_variance()

            return log_likelihood_and_variance

        def log_likelihood(parameters):
            likelihood.parameters.update(parameters)
            return likelihood.log_likelihood_ratio()

        return log_likelihood

    def likelihood_extras(self, likelihood, parameters):
        likelihood.parameters.update(parameters)
        likelihood.parameters, added_keys = likelihood.conversion_function(
            likelihood.parameters,
        )
        likelihood.hyper_prior.parameters.update(parameters)

        log_bayes_factors, variances = \
            likelihood._compute_per_event_ln_bayes_factors()

        detection_efficiency, detection_variance = \
            likelihood.selection_function.detection_efficiency(parameters)

        selection = -likelihood.n_posteriors * jnp.log(detection_efficiency)
        selection_variance = (
            likelihood.n_posteriors ** 2
            * detection_variance
            / detection_efficiency ** 2
        )

        log_likelihood = jnp.sum(log_bayes_factors) + selection
        variance = jnp.sum(variances) + selection_variance

        return dict(
            log_likelihood=log_likelihood,
            variance=variance,
            log_bayes_factors=log_bayes_factors,
            variances=variances,
            detection_efficiency=detection_efficiency,
            detection_variance=detection_variance,
            selection=selection,
            selection_variance=selection_variance,
        )

    def trainer(
        self,
        key,
        vmap=True,
        flow=None,
        batch_size=None,
        steps=None,
        learning_rate=None,
        optimizer=None,
        taper=None,
        temper_schedule=None,
        **tqdm_kwargs,
    ):
        print('GWAX - getting ready...')

        names = tuple(self.prior_bounds.keys())
        bounds = tuple(self.prior_bounds.values())
        prior = self.get_prior(bounds)

        _log_likelihood_and_variance = self.get_log_likelihood(self.likelihood, True)
        if vmap:
            log_likelihood_and_variance = jax.vmap(_log_likelihood_and_variance)
        else:
            log_likelihood_and_variance = lambda parameters: jax.lax.map(
                _log_likelihood_and_variance, parameters,
            )

        if taper is None:
            taper = lambda variance: 0.0

        def log_target(samples):
            parameters = dict(zip(names, samples.T))
            log_lkls, variances = log_likelihood_and_variance(parameters)
            return prior.log_prob(samples) + log_lkls + taper(variances)

        if flow is None:
            key, _key = jax.random.split(key)
            flow = Flow(bounds=bounds).default_flow(_key)  

        params, static = equinox.partition(
            pytree=flow,
            filter_spec=equinox.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        def loss_fn(params, key, step):
            flow = equinox.combine(params, static)
            samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
            log_targets = log_target(samples) * temper_schedule(step)
            return jnp.mean(log_flows - log_targets)

        if optimizer is None:
            optimizer = optax.adam
        if callable(optimizer):
            optimizer = optimizer(learning_rate)

        state = optimizer.init(params)

        if temper_schedule is None:
            temper_schedule = lambda step: 1.0

        tqdm_defaults = dict(
            print_rate=1,
            tqdm_type='auto',
            desc='GWAX - variational training',
        )
        for arg in tqdm_kwargs:
            tqdm_defaults[arg] = tqdm_kwargs[arg]

        @jax_tqdm.scan_tqdm(steps, **tqdm_defaults)
        @equinox.filter_jit
        def update(carry, step):
            key, params, state = carry
            key, _key = jax.random.split(key)
            loss, grad = equinox.filter_value_and_grad(loss_fn)(params, _key, step)
            updates, state = optimizer.update(grad, state, params)
            params = equinox.apply_updates(params, updates)
            return (key, params, state), loss

        print('GWAX - JAX jitting...')
        t0 = time.time()
        (key, params, state), losses = jax.lax.scan(
            update, (key, params, state), jnp.arange(steps),
        )
        flow = equinox.combine(params, static)
        print(f'GWAX: Total time = {time.time() - t0} s')

        return flow, losses
    












import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class GaussianMixtureLikelihood:
    """
    Log-likelihood for a fixed Gaussian mixture 

    Args
    ----
    means        : (K, D) array component means 
    covs         : (K, D, D) array component covariances (positive semi-definite)
    weights      : (K,) array of normalised weights 
    """

    def __init__(self, means, covs, weights):
        self.means    = jnp.asarray(means)
        self.covs     = jnp.asarray(covs)
        # putting weights in simplex so that they sum up to 1
        self.log_w    = jax.nn.log_softmax(jnp.asarray(weights))
        # K = nr of components, D = nr of dimensions
        self.K, self.D = self.means.shape
        self.parameters = {f"x{i}": 0.0 for i in range(self.D)}

        # normalising constant of a multivariate Gaussian (pdf integrate to 1)
        self.cov_invs  = jax.vmap(jnp.linalg.inv)(self.covs)                          # (K, D, D)
        self.log_dets  = jax.vmap(lambda C: jnp.linalg.slogdet(C)[1])(self.covs)      # (K,)
        # function
        self.log_norms = -0.5 * (self.D * jnp.log(2.0 * jnp.pi) + self.log_dets)      # (K,)


    def update(self, new_params):
        """Current evaluation sample x."""
        self.parameters = new_params

    def ln_likelihood_and_variance(self):
        """Return log p(x) and a dummy variance. It was given by definition"""
        x = jnp.array(list(self.parameters.values()))
        logp = self._log_prob(x)
        var  = 0.0  
        return logp, var

    
    #  log-pdf                                                             
    def _log_prob(self, x):
        """
        Returns log function for gaussian mixture
        """
        def one_component(mu, cov_inv, log_norm):
            diff = x - mu
            quad = diff @ cov_inv @ diff
            return log_norm - 0.5 * quad                         
        
        # make a mixture
        log_comp = jax.vmap(one_component)(self.means,
                                           self.cov_invs,
                                           self.log_norms)
        log_mix  = logsumexp(self.log_w + log_comp)              
        return log_mix












import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner


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
                        smooth=1.0, 
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


class GaussianMixtureGenerator:
    """Utility class to generate samples from a mixture of Gaussians."""

    @staticmethod
    def generate_gaussian_mixture(n_dim: int,
                                  n_gaussians: int = 3,
                                  n_samples: int = 10_000,
                                  means: list = None,
                                  covariances: list = None,
                                  weights: list = None,
                                  width_mean: float = 12.0,
                                  width_cov: float = 4.0):
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
                #print("this_means")
                print(this_means)
                
                means.append(this_means)
        #print(f"Means: {means}")
            
        # If no covariance matrix is given, generate identity matrices
        if covariances is None:
            covariances = []
            for _ in range(n_gaussians):
                jax_key, subkey = jax.random.split(jax_key)
                A = jax.random.uniform(subkey, (n_dim, n_dim), minval=-width_cov, maxval=width_cov)
                B = jnp.dot(A, A.transpose())
                covariances.append(B)
        #print(f"Covariances: {covariances}")
        
        # If no weights are given, use equal weights between the Gaussians
        if weights is None:
            weights = [1.0 / n_gaussians] * n_gaussians
        #print(f"Weights: {weights}")
            
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
        return samples, means, covariances, weights
    



class Runner:
    """
    Executes the full VI experiment and stores results.

    Arguments
    ---------
    base_results_dir : str, optional
        Root directory where numbered `VI_results_*` subfolders will be
        created.  Default is "./results".
    save : bool, optional
        If False the run is executed but nothing is written.
    """
    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    import os, re

    @staticmethod
    def get_next_available_outdir(base_dir: str, prefix: str = "VI_results") -> str:
        """
        Create and return a unique output directory
        """
        import os, re
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        existing = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        matches = [re.match(rf"{prefix}_(\d+)", name) for name in existing]
        numbers = [int(m.group(1)) for m in matches if m]
        next_number = max(numbers, default=0) + 1

        unique_dir = os.path.join(base_dir, f"{prefix}_{next_number}")
        os.makedirs(unique_dir)
        print(f"Using output directory: {unique_dir}")
        return unique_dir

    
    # main experiment
    def __init__(self, base_results_dir: str = "./results", save: bool = True):
        
        import jax.numpy as jnp
        from jax import random
        import optax
        import numpy as np
        import corner
        import matplotlib.pyplot as plt
        import os, re                                                             
        

        # 
        self.outdir = None
        if save:
            self.outdir = self.get_next_available_outdir(base_results_dir)

        # 
        np.random.seed(2)

        # true samples and VI parameters
        true_samples, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
            n_dim=3, n_gaussians=3
        )

        vi_means   = jnp.asarray(means)
        vi_covs    = jnp.asarray(covariances)
        vi_weights = jnp.asarray(weights)

        print(vi_means.shape)
        print(vi_covs.shape)
        print(vi_weights.shape)

        key = random.key(0)

        likelihood = GaussianMixtureLikelihood(
            vi_means,
            vi_covs,
            vi_weights,
        )

        prior_bounds = {
            "x0": [-30.0, 30.0],
            "x1": [-30.0, 30.0],
            "x2": [-30.0, 30.0],
        }

        # cosine tempering schedule
        def cosine_temper(step, *, total_steps, beta_min=0.49):
            t = step / total_steps
            return beta_min + 0.5 * (1 - beta_min) * (1 + jnp.cos(jnp.pi * t))

        steps = 2000
        temper_schedule = lambda s: cosine_temper(s, total_steps=steps, beta_min=0.7)

        # optimiser
        lr_sched  = optax.cosine_decay_schedule(0.05, steps)
        optimizer = optax.adam(lr_sched)

        vi  = VI(prior_bounds=prior_bounds, likelihood=likelihood)
        key = random.key(0)

        flow, losses = vi.trainer(
            key=key,
            batch_size=1000,
            steps=steps,
            optimizer=optimizer,
            temper_schedule=temper_schedule,
        )

        vi_samples = np.array(flow.sample(random.key(1), (10_000,)))
        print("Number of samples:", len(vi_samples))
        print("Samples:", vi_samples[0:5])

        #  corner plot 
        vi_np   = np.asarray(vi_samples)
        true_np = np.asarray(true_samples)

        hist_kwargs = {"color": "red", "density": True}
        fig = corner.corner(
            vi_np,
            color="red",
            label="VI Approximation",
            hist_kwargs=hist_kwargs,
            show_titles=True,
        )

        hist_kwargs = {"color": "blue", "density": True}
        corner.corner(
            true_np,
            fig=fig,
            color="blue",
            label="True Normal",
            hist_kwargs=hist_kwargs,
            show_titles=True,
        )

        plt.legend()

        # 
        if self.outdir is not None:
            plot_path = os.path.join(self.outdir, "vi_vs_true_corner.png")
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Corner plot saved to {plot_path}")
        else:
            plt.show()
        plt.close(fig)

        #  sample statistics 
        np.set_printoptions(precision=4, suppress=True)
        self.pm = vi_samples.mean(axis=0)
        self.pv = vi_samples.var(axis=0)
        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)

        stats_str = (
            "This is pm (mean of VI samples):\n" + str(self.pm) +
            "\n\nThis is pv (variance of VI samples):\n" + str(self.pv) +
            "\n\nThis is qm (mean of true samples):\n" + str(self.qm) +
            "\n\nThis is qv (variance of true samples):\n" + str(self.qv) + "\n"
        )

        if self.outdir is not None:
            stats_path = os.path.join(self.outdir, "sample_statistics.txt")
            with open(stats_path, "w") as f:
                f.write(stats_str)
            print(f"Sample statistics saved to {stats_path}")
        else:
            print(stats_str)

        #  KL divergence
        def gau_kl(pm, pv, qm, qv):
            if (len(qm.shape) == 2):
                axis = 1
            else:
                axis = 0
            dpv = pv.prod()
            dqv = qv.prod(axis)
            iqv = 1. / qv
            diff = qm - pm
            return 0.5 * (
                np.log(dqv / dpv)
                + (iqv * pv).sum(axis)
                + (diff * iqv * diff).sum(axis)
                - len(pm)
            )

        kl_div = gau_kl(self.pm, self.pv, self.qm, self.qv)
        print(f"KL divergence between VI approximation and True Normal: {kl_div:.4f} nats")

        if self.outdir is not None:
            kl_path = os.path.join(self.outdir, "kl_divergence.txt")
            with open(kl_path, "w") as f:
                f.write(f"KL divergence (nats): {kl_div:.6f}\n")
            print(f"KL divergence saved to {kl_path}")

        
        self.kl_div   = kl_div
        self.vi_samples = vi_samples
        self.true_samples = true_samples



if __name__ == "__main__":
    Runner()  


