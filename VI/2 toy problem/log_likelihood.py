
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
