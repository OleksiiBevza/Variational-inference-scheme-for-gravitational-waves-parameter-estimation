from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox

from flowjax.distributions import (
    Normal,
    VmapMixture,
    StandardNormal,          
    Transformed,
)

from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Stack,
    Tanh,
)

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
    
    def default_flow(
        self,
        key,
        *,
        n_components: int = 4,          
        weights: jnp.ndarray = jnp.array([0.25, 0.25, 0.25, 0.25]),  
        **kwargs,
        ):
        """
        VmapMixture allows for a mixture of Gaussians with multiple components in flow architecture. 
        Uses a Cholesky-based quadratic form for numerical stability.

        Parameters to be specified for distribution:
        n_components: = number of components in the mixture of Gaussians
        weights = array of starting point weights fot the mixture of Gaussians (initial weights are trainable)

        Parameters to be specified for flow training:
        nn_depth=1 by default,
        nn_block_dim=8 by default,
        flow_layers=1 by default
        """
        dim = len(self.bounds)

        # a set of independent N(0,1) components
        # locs and scales are (K, D) arrays with 0's and 1's.
        locs   = jnp.zeros((n_components, dim))
        scales = jnp.ones((n_components, dim))
        normal_bank = eqx.filter_vmap(Normal)(locs, scales) 

               
        # set into a mixture distribution
        base_dist = VmapMixture(normal_bank, weights=weights)
        
        # flow parameters
        flow_defaults = dict(
            key=key,
            base_dist=base_dist,
            invert=False,
            nn_depth=1,
            nn_block_dim=8,
            flow_layers=1,
        )
        flow_defaults.update(kwargs)  
        flow = block_neural_autoregressive_flow(**flow_defaults)

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
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from typing import Mapping, Sequence, Union


class GaussianMixtureLikelihood:
    """
    Log-likelihood or log-pdf for a multi-component Gaussian mixture
    Parameters:
    means: an array in shape (K, D)
        The component mean vectors μ_k.
    covs : an array in shape (K, D, D), full, positive-definite covariance matrices.
    weights: an array in shape (K,) as mixture weights. Either:
        (i)  probabilities that sum up to 1;
        (ii) unnormalised logits if `logits=True`.
    logits: bool, by default `False`
        (i)  if True, `weights` are logits and converted to probabilities using softmax.  
        (ii) if False, `weights` are probabilities.
    _eps: float, by default given 1e-30 as an offset to avoid log(0).
    """


    def __init__(
        self,
        means:   jnp.ndarray,
        covs:    jnp.ndarray,
        weights: jnp.ndarray,
        *,
        logits: bool = False,
        _eps: float = 1e-30,
    ):
        # store components parameters (means and covariances) 
        self.means = jnp.asarray(means)            # (K, D)
        self.covs  = jnp.asarray(covs)             # (K, D, D)
        self.K, self.D = self.means.shape          # number of components (K) and dimensions (D)

        # putting weights in simplex so that they sum up exctly to 1
        if logits:                                 
            self.log_w = jax.nn.log_softmax(jnp.asarray(weights))   # (K,)
        else:                                     
            w = jnp.asarray(weights)
            w = w / jnp.sum(w)                     
            self.log_w = jnp.log(w + _eps)         # avoid log(0) by adding _eps 

        # Cholesky factors
        self.chols = jax.vmap(jnp.linalg.cholesky)(self.covs)       # (K, D, D)
        self.log_dets = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(self.chols, axis1=-2, axis2=-1)),
            axis=-1,
        )  # (K,)

        # multivariate function
        self.log_norms = -0.5 * (self.D * jnp.log(2.0 * jnp.pi) + self.log_dets)  # (K,)
        # formatting for a variational inference trainer
        self.parameters: Mapping[str, float] = {f"x{i}": 0.0 for i in range(self.D)}



    # provided by definition for variational inference trainer
    def update(self, new_params: Mapping[str, float]) -> None:
        """Replace `self.parameters`"""
        self.parameters = new_params

    # provided by definition for variational inference trainer
    def ln_likelihood_and_variance(self) -> tuple[jnp.ndarray, float]:
        """
        Return log p(x) with variance (0.0),
        which is expected by VI class.
        """
        # build a dimension vector D in order "x0" … "x{D-1}" 
        x = jnp.array(list(self.parameters.values()))  # (D,)
        return self._log_prob_single(x), 0.0

    

    # log-pdf for a single point                                     
    def _log_prob_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates given point x, adds weight of the component, and returns the log-probability that the mixture would generate x.
        Compute log p(x) for a single point x (shape (D,)).
        Uses a Cholesky-based quadratic form for numerical stability.
        """
        # find distance between point x and means od each component
        diffs = x - self.means   # (K, D)

        # use Cholesky factorization (finding offsets for a particular Gaussian component)
        def quad_form(diff, L):
            #
            y = solve_triangular(L, diff, lower=True)
            return jnp.sum(y**2)                     

        quad = jax.vmap(quad_form)(diffs, self.chols)      

        # convert distance into log-density of each component
        log_comp = self.log_norms - 0.5 * quad               

        # mix components with weights to get the mixture log density and return
        return logsumexp(self.log_w + log_comp)               
  
    
    def log_prob(
        self,
        xs: Union[jnp.ndarray, Sequence[jnp.ndarray]],
    ) -> jnp.ndarray:
        """
        Evaluate log likelihood for given array of points.

        Parameters:
        x's: array_like, shape (..., D), a single point (D,) or a batch of points.

        Returns:
        logp: jnp.ndarray, shape (...,)
            Log-likelihoods with the same shape as x's.
        """
        xs = jnp.asarray(xs)
        flat = xs.reshape(-1, self.D)                       
        flat_logp = jax.vmap(self._log_prob_single)(flat)   
        return flat_logp.reshape(xs.shape[:-1])         



















import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner

"""
Sourse: https://github.com/ThibeauWouters?tab=repositories
"""

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
                        smooth=0.5, 
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
                                  n_gaussians: int = 2,
                                  n_samples: int = 10_000,
                                  means: list = None,
                                  covariances: list = None,
                                  weights: list = None,
                                  width_mean: float = 10.0,
                                  width_cov: float = 3.0):
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
    Executes the VI experiment and stores results in a folder results.       
    """
    # ------------------------------------------------------------------
    # saving results of an experiment
    # ------------------------------------------------------------------
    import os, re

    @staticmethod
    def get_next_available_outdir(base_dir: str, prefix: str = "toy2_results") -> str:
        """
        Creates and returns a unique output directory
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
    

    # ---------------------------------------------------------------------
    # PARAMETRIC AND NON-PARAMETRIC KL DIVERGENCE ESTIMATORS 
    # ---------------------------------------------------------------------
    import numpy as np, warnings, os
    from typing import Tuple

    # Sourse: https://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
    @staticmethod
    def gau_kl(pm: np.ndarray, pv: np.ndarray,
               qm: np.ndarray, qv: np.ndarray) -> float:
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
         of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)                 # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)            # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis)   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)                         # - N
        ))
    
    # Non-parametric KL divergence estimators
    # Sourse: https://github.com/nhartland/KL-divergence-estimators
    @staticmethod
    def _knn_distance(point: np.ndarray, sample: np.ndarray, k: int) -> float:
        """Euclidean distance from `point` to it's `k`-Nearest
        Neighbour in `sample`
        This function works for points in arbitrary dimensional spaces.
        """
        # Compute all euclidean distances
        norms = np.linalg.norm(sample - point, axis=1)
        # Return the k-th nearest
        return np.sort(norms)[k]

    @staticmethod
    def _verify_sample_shapes(s1: np.ndarray, s2: np.ndarray, k: int):
        # Expects [N, D]
        assert len(s1.shape) == len(s2.shape) == 2 
        # Check dimensionality of sample is identical
        assert s1.shape[1] == s2.shape[1] 

    @staticmethod
    def naive_estimator(s1: np.ndarray, s2: np.ndarray, k: int) -> float:
        """KL-Divergence estimator using brute-force (numpy) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
        """
        Runner._verify_sample_shapes(s1, s2, k)
        n, m = len(s1), len(s2)
        D = np.log(m / (n - 1))
        d = float(s1.shape[1])
        for p1 in s1:
            nu  = Runner._knn_distance(p1, s2, k - 1)   # -1 because 'p1' is not in 's2'
            rho = Runner._knn_distance(p1, s1, k)
            D  += (d / n) * np.log(nu / rho)
        return D

    @staticmethod
    def scipy_estimator(s1: np.ndarray, s2: np.ndarray, k: int) -> float:
        """KL-Divergence estimator using scipy's KDTree
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
        """
        Runner._verify_sample_shapes(s1, s2, k)
        from scipy.spatial import KDTree
        n, m = len(s1), len(s2)
        d = float(s1.shape[1])
        D = np.log(m / (n - 1))
        nu_d, nu_i = KDTree(s2).query(s1, k)
        rho_d, rhio_i = KDTree(s1).query(s1, k + 1)
        # KTree.query returns different shape in k==1 vs k > 1
        if k > 1:
            D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
        else:
            D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))
        return D

    @staticmethod
    def skl_estimator(s1: np.ndarray, s2: np.ndarray, k: int) -> float:
        """KL-Divergence estimator using scikit-learn's NearestNeighbours
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
        """
        Runner._verify_sample_shapes(s1, s2, k)
        from sklearn.neighbors import NearestNeighbors
        n, m = len(s1), len(s2)
        d = float(s1.shape[1])
        D = np.log(m / (n - 1))
        s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1).fit(s1)
        s2_neighbourhood = NearestNeighbors(n_neighbors=k).fit(s2)
        for p1 in s1:
            s1_distances, _ = s1_neighbourhood.kneighbors([p1], k + 1)
            s2_distances, _ = s2_neighbourhood.kneighbors([p1], k)
            rho = s1_distances[0][-1]
            nu = s2_distances[0][-1]
            D += (d / n) * np.log(nu / rho)
        return D


    @staticmethod
    def skl_efficient(s1: np.ndarray, s2: np.ndarray, k: int) -> float:
        """An efficient version of the scikit-learn estimator by @LoryPack
           s1: (N_1,D) Sample drawn from distribution P
           s2: (N_2,D) Sample drawn from distribution Q
           k: Number of neighbours considered (default 1)
           return: estimated D(P|Q)

           Contributed by Lorenzo Pacchiardi (@LoryPack)
        """
        Runner._verify_sample_shapes(s1, s2, k)
        from sklearn.neighbors import NearestNeighbors
        import warnings
        n, m = len(s1), len(s2)
        d = float(s1.shape[1])
        s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
        s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)
        s1_distances, _ = s1_neighbourhood.kneighbors(s1, k + 1)
        s2_distances, _ = s2_neighbourhood.kneighbors(s1, k)
        rho = s1_distances[:, -1]
        nu = s2_distances[:, -1]
        if np.any(rho == 0):
            warnings.warn(
                f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
                f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
                f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
                RuntimeWarning,
            )
        D = np.sum(np.log(nu / rho))
        return (d / n) * D + np.log(m / (n - 1)) # this second term should be enough for it to be valid for m \neq n



    # write down all KL metrics 
    def kl_metrics(self, n_samples: int = 10_000, k: int = 10,
                   outdir: str | None = None,
                   filename: str = "kl_metrics.txt") -> None:
        """
        Saves parametric and k‑NN KL estimates to <outdir>/<filename>.
        """
        import os
        outdir = outdir or self.outdir
        if outdir is None:
            raise ValueError("No output directory specified.")

        # parametric stats
        pm = self.vi_samples.mean(axis=0)
        pv = self.vi_samples.var(axis=0)
        qm = self.true_samples.mean(axis=0)
        qv = self.true_samples.var(axis=0)
        kl_div = self.gau_kl(pm, pv, qm, qv)

        # sample for k‑NN
        s1 = self.vi_samples[:n_samples]
        s2 = self.true_samples[:n_samples]

        naive      = self.naive_estimator(s1, s2, k)
        scipy_val  = self.scipy_estimator(s1, s2, k)
        skl_val    = self.skl_estimator(s1, s2, k)
        skl_e_val  = self.skl_efficient(s1, s2, k)

        out_path = os.path.join(outdir, filename)
        with open(out_path, "w") as f:
            f.write(f"Parametric KL: {kl_div:.8f}\n")
            f.write(f"KNN naive KL: {naive:.8f}\n")
            f.write(f"KNN scipy KL: {scipy_val:.8f}\n")
            f.write(f"KNN sklearn KL: {skl_val:.8f}\n")
            f.write(f"KNN sklearn fast KL: {skl_e_val:.8f}\n")
        print(f"KL metrics saved to {out_path}")





    
    # main experiment
    def __init__(self, base_results_dir: str = "./results", save: bool = True):
        
        import jax.numpy as jnp
        from jax import random
        import optax
        import numpy as np
        import corner
        import matplotlib.pyplot as plt
        import os
        import re
        import json 

        import matplotlib
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['DejaVu Serif']   
        matplotlib.rcParams['text.usetex'] = False             
                                                           
        
        # saving the results
        self.outdir = None
        if save:
            self.outdir = self.get_next_available_outdir(base_results_dir)





        # ---------------------------------------------------------------------------------
        # DEFINING PARAMETERS FOR VI SAMPLER
        # ---------------------------------------------------------------------------------
        np.random.seed(900)

        # generate true samples from GaussianMixtureGenerator class and parameters (means, covariances and weights) 
        # which will be used in the likelihood function for VI trainer
        true_samples, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
            n_dim=2, n_gaussians=2, weights = [0.5, 0.5],
        )
         

        vi_means   = jnp.asarray(means)
        vi_covs    = jnp.asarray(covariances)
        vi_weights = jnp.asarray(weights)

        #print(vi_means.shape)
        #print(vi_covs.shape)
        #print(vi_weights.shape)

        key = random.key(123)
        # define likelihood function for VI trainer 
        likelihood = GaussianMixtureLikelihood(
            vi_means,
            vi_covs,
            vi_weights,
        )

        # define prior bounds for VI trainer
        prior_bounds = {
            "x0": [-30.0, 30.0],
            "x1": [-30.0, 30.0],
            #"x2": [-15.0, 15.0],
            #"x3": [-15.0, 15.0],
            #"x4": [-15.0, 15.0],
            #"x5": [-15.0, 15.0],
            #"x6": [-15.0, 15.0],
            #"x7": [-15.0, 15.0],
            #"x8": [-15.0, 15.0],
            #"x9": [-15.0, 15.0],
            #"x10": [-15.0, 15.0],
            #"x11": [-15.0, 15.0],
            #"x12": [-15.0, 15.0],
            #"x13": [-15.0, 15.0],
            #"x14": [-15.0, 15.0],
       
        }

        # cosine tempering schedule function
        def cosine_temper(step, *, total_steps, beta_min=0.49):
            t = step / total_steps
            return beta_min + 0.5 * (1 - beta_min) * (1 + jnp.cos(jnp.pi * t))

        steps = 20000
        temper_schedule = lambda s: cosine_temper(s, total_steps=steps, beta_min=0.7)

        # optimiser
        lr_sched  = optax.cosine_decay_schedule(0.065, steps)
        optimizer = optax.adam(lr_sched)



        # ---------------------------------------------------------------------------------
        # DEFINING VI TRAINER
        # ---------------------------------------------------------------------------------      
        vi  = VI(prior_bounds=prior_bounds, likelihood=likelihood)
        key = random.key(0)

        # defining VI trainer
        flow, losses = vi.trainer(
            key=key,
            batch_size=2000,
            steps=steps,
            optimizer=optimizer,
            temper_schedule=temper_schedule,
        )

        # extract VI samples from VI trainer
        vi_samples = np.array(flow.sample(random.key(1), (10_000,)))
        #print("Number of samples:", len(vi_samples))
        #print("Samples:", vi_samples[0:5])

        # saving VI samples results and true samples from GaussianMixtureGenerator class to json files
        if self.outdir:
            # VI samples
            vi_path = os.path.join(self.outdir, "vi_samples.json")
            with open(vi_path, "w") as f:
                json.dump(vi_samples.tolist(), f)
            print(f"VI samples saved to {vi_path}")

            # true samples
            true_path = os.path.join(self.outdir, "true_samples.json")
            with open(true_path, "w") as f:
                json.dump(true_samples.tolist(), f)
            print(f"True samples saved to {true_path}")






        # --------------------------------------------------------------------------------
        # VISUAL INSPECTION OF THE RESULTS
        # --------------------------------------------------------------------------------
        
        # loss plot function
        fig_loss = plt.figure(figsize=(8, 4))
        plt.plot(jnp.log(losses))
        plt.ylim(0, 10)
        plt.xlabel("Training step")
        plt.ylabel("VI log(loss)")
        plt.xticks(rotation=45)  
        plt.tight_layout()

        # saving corner plot into the output directory
        if self.outdir:
            loss_path = os.path.join(self.outdir, "loss_function.png")
            fig_loss.savefig(loss_path, bbox_inches="tight")
            print(f"Loss function saved to {loss_path}")
        else:
            plt.show()
        plt.close(fig_loss)



        #  corner plot 
        vi_np   = np.asarray(vi_samples)
        true_np = np.asarray(true_samples)

        #print("vi_np shape   :", vi_np.shape)
        #print("true_np shape :", true_np.shape)

        hist_kwargs = {"color": "blue", "density": True}
        fig = corner.corner(
            vi_np,
            color="blue",
            label="VI Approximation",
            hist_kwargs=hist_kwargs,
            show_titles=True,
        )

        hist_kwargs = {"color": "red", "density": True}
        corner.corner(
            true_np,
            fig=fig,
            color="red",
            label="True Normal",
            hist_kwargs=hist_kwargs,
            show_titles=True,
        )

        handles = [
            plt.Line2D([], [], color="blue", label="VI Approximation"),
            plt.Line2D([], [], color="red", label="True Normal"),
        ]
        plt.legend(handles=handles, loc="upper right")

        
        # saving corner plot into the output directory
        if self.outdir is not None:
            plot_path = os.path.join(self.outdir, "vi_vs_true_corner.png")
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Corner plot saved to {plot_path}")
        else:
            plt.show()

        plt.close(fig)



        #  sample statistics (mean and variance of VI and true samples)
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
        
        
        self.vi_samples = vi_samples
        self.true_samples = true_samples


# running the whole experiment
if __name__ == "__main__":
    runner = Runner()          
    runner.kl_metrics()   




