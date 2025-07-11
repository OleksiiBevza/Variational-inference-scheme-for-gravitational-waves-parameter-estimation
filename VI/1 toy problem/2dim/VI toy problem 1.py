
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

"""Sourse: https://github.com/mdmould/gwax/blob/main/gwax/flows.py """
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
            nn_depth=5,
            nn_block_dim=124,
            flow_layers=5,
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

"""Sourse: https://github.com/mdmould/gwax/blob/main/gwax/flows.py """
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
    



# packages for diagnostics
import optax
import jax
from jax import random
import numpy as np
import corner 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import jax.numpy as jnp

class MultivariateNormalLikelihood:
    def __init__(self, dim, mean, cov):
        """
        Multivariate normal likelihood with specified mean and covariance.

        Args:
            dim: Dimension of the distribution
            mean: Optional mean vector (if None, random)
            cov: Optional full covariance matrix (if None, random diagonal)
        """
        self.dim = dim
        self.mean = mean
        self.cov = cov
        self.parameters = {f"x{i}": 0.0 for i in range(dim)}

    def update(self, new_params):
        self.parameters = new_params

    def ln_likelihood_and_variance(self):
        x = jnp.array(list(self.parameters.values()))
        log_likelihood = self._log_prob(x)
        variance = 0.0* jnp.trace(self.cov)     # as for now it is all here 
        return log_likelihood, variance

    def _log_prob(self, x):
        diff = x - self.mean
        cov_inv = jnp.linalg.inv(self.cov)
        log_det_cov = jnp.linalg.slogdet(self.cov)[1]
        quad_form = diff @ cov_inv @ diff
        return -0.5 * (self.dim * jnp.log(2 * jnp.pi) + log_det_cov + quad_form)

    def sample(self, size=None):
        return np.random.multivariate_normal(self.mean, self.cov, size=size)
    









import numpy as np
import jax
from jax import random
import optax
from scipy.stats import multivariate_normal
import numpy as np

import corner
import matplotlib.pyplot as plt


import warnings

import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import os
import re
import json

class Runner:
    def __init__(
        self,
        dim=15,
        mean=None,
        cov=None,
        prior_bounds=None,
        steps=10000,
        batch_size=2000,
        learning_rate=0.065,
        seed=0
    ):
        self.dim = dim
        self.mean = np.array([0.0, 2.0, 4.0, -5.0, -3.0, -1.0, 1.0, 5.0, 8.0, 10.0, 12.0, -7.0, -4.0, -11.0, -13.0])
        self.seed = seed
        self.key = random.key(seed)
        self.steps = steps
        self.batch_size = batch_size
        self.flow = None
        self.losses = None
        self.vi_samples = None
        self.true_samples = None

        np.random.seed(seed)
        if cov is not None:
            self.cov = cov
        else:
            # random positive semi-definite matrix
            A = np.random.uniform(-1.0, 1.0, dim ** 2).reshape((dim, dim))
            self.cov = np.dot(A, A.transpose())

        self.prior_bounds = dict(
            x0  = [-5.5, 5.5],
            x1  = [-3.5, 7.5],
            x2  = [-1.5, 9.5],
            x3  = [-10.5, 0.5],
            x4  = [-8.5, 2.5],
            x5  = [-6.5, 4.5],
            x6  = [-4.5, 6.5],
            x7  = [-0.5, 10.5],
            x8  = [2.5, 13.5],
            x9  = [4.5, 15.5],
            x10 = [6.5, 17.5],
            x11 = [-12.5, -1.5],
            x12 = [-9.5, 1.5],
            x13 = [-16.5, -5.5],
            x14 = [-18.5, -7.5],
        )


        #
        self.learning_rate = optax.cosine_decay_schedule(learning_rate, steps)
        self.optimizer = optax.adam(self.learning_rate)

        # Instantiate likelihood and VI
        self.likelihood = MultivariateNormalLikelihood(self.dim, self.mean, self.cov)
        self.vi = VI(prior_bounds=self.prior_bounds, likelihood=self.likelihood)

        self.flow = None
        self.losses = None



    @staticmethod
    def get_next_available_outdir(base_dir: str, prefix: str = "VI_results") -> str:
        """
        Function makes folder "results" and stores the outcomes of the experiments in the folder. 
        Every result will have its consecutive number: vi_results_1 .... vi_results_n

        Args:
        base_dir (str): path to folder "results".
        prefix (str): subdirectory names. Defaults to 'VI_results'.

        Returns:
        str:  path to folder.
        """


        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        matches = [re.match(rf"{prefix}_(\d+)", name) for name in existing]
        numbers = [int(m.group(1)) for m in matches if m]
        next_number = max(numbers, default=0) + 1
        unique_dir = os.path.join(base_dir, f"{prefix}_{next_number}")
        os.makedirs(unique_dir)
        print(f"Using output directory: {unique_dir}")
        return unique_dir


    def run(self):
        print("Starting VI training...")
        self.key = random.key(self.seed)  # reset key for reproducibility
        self.flow, self.losses = self.vi.trainer(
            key=self.key,
            batch_size=self.batch_size,
            steps=self.steps,
            optimizer=self.optimizer,
            temper_schedule = lambda step: self.cosine_temper(step,
                                                              beta_min=0.49),
        )
        print("Training finished!")
        return self.flow, self.losses


    def cosine_temper(self, step, *, beta_min=0.49):
        """
        Cosine schedule on [0, steps]:
        β(0) = 1  and  β(steps) = beta_min  (e.g. 0.7).
        """
        t = step / self.steps               
        return beta_min + 0.5 * (1 - beta_min) * (1 + jnp.cos(jnp.pi * t))


    def get_vi_samples(self, n_samples=10_000, outdir=None):
        if self.vi_samples is None or len(self.vi_samples) != n_samples:
            self.vi_samples = np.array(self.flow.sample(jax.random.key(1), (n_samples,)))

        # 
        if outdir is not None:
            samples_list = self.vi_samples.tolist()
            out_path = os.path.join(outdir, "vi_samples.json")
            with open(out_path, "w") as f:
                json.dump(samples_list, f)
            print(f"VI samples saved to {out_path}")
    
        
        return self.vi_samples



    def get_true_samples(self, n_samples=10_000, outdir=None):
        if self.true_samples is None or len(self.true_samples) != n_samples:
            mean1 = np.array(self.likelihood.mean)
            cov1 = np.array(self.likelihood.cov)
            true_dist = multivariate_normal(mean=mean1, cov=cov1, allow_singular=True)
            self.true_samples = true_dist.rvs(size=n_samples, random_state=1)
        
        # 
        if outdir is not None:
            samples_list1 = self.true_samples.tolist()
            out_path = os.path.join(outdir, "true_samples.json")
            with open(out_path, "w") as f:
                json.dump(samples_list1, f)
            print(f"True samples saved to {out_path}")
        
        return self.true_samples



    #def print_vi_samples(self, n_samples=10_000):
        #vi_samples = self.get_vi_samples(n_samples)
        #print(f"\nNumber of VI samples: {len(vi_samples)}")
        #print("First 5 samples:\n", vi_samples[0:5])

    #def print_true_samples(self, n_samples=10_000):
        #true_samples = self.get_true_samples(n_samples)
        #print(f"\nNumber of TRUE samples: {len(true_samples)}")
        #print("First 5 TRUE samples:\n", true_samples[0:5])



    def plot_losses(self, hline=1.7, figsize=(10, 6), outdir=None):
        if self.losses is None:
            print("No losses to plot. Have you run training?")
            return
        plt.figure(figsize=figsize)
        plt.plot(self.losses)
        plt.axhline(hline, c='r')
        plt.xlim(-500, min(self.steps, 1200))
        plt.ylim(0, max(1000, np.max(self.losses)))
        plt.xlabel("Training step")
        plt.ylabel("VI loss")
        plt.title("VI Loss Curve")
    
        # save results **before** showing or closing
        if outdir is not None:
            plot_path = os.path.join(outdir, "vi_loss_curve.png")
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"VI loss curve plot saved to {plot_path}")
        
        #plt.show()
        plt.close()



    def plot_vi_vs_true(self, n_samples=10_000, outdir=None):
        vi_samples = self.get_vi_samples(n_samples)
        true_samples = self.get_true_samples(n_samples)
        # VI samples in blue
        hist_kwargs = {"color": "blue", "density": True}
        fig = corner.corner(vi_samples, color="blue", label="VI Approximation",
                            hist_kwargs=hist_kwargs, show_titles=True)
        # true samples in red
        hist_kwargs = {"color": "red", "density": True}
        corner.corner(true_samples, fig=fig, color="red", label="True Normal",
                      hist_kwargs=hist_kwargs, show_titles=True)
        # 
        handles = [
            plt.Line2D([], [], color="blue", label="VI Approximation"),
            plt.Line2D([], [], color="red", label="True Normal"),
        ]
        plt.legend(handles=handles, loc="upper right")

                
        if outdir is not None:
            plot_path = os.path.join(outdir, "vi_vs_true_corner.png")
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Corner plot saved to {plot_path}")
        # plt.show()  
        plt.close(fig) 
        #plt.show()



    def print_statistics(self, n_samples=10_000, outdir=None, filename="sample_statistics.txt"):
        vi_samples = self.get_vi_samples(n_samples)
        true_samples = self.get_true_samples(n_samples)

        self.pm = vi_samples.mean(axis=0)
        self.pv = vi_samples.var(axis=0)
        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)
    
   

        np.set_printoptions(precision=4, suppress=True)

        stats_str = (
            "This is pm (mean of VI samples):\n" + str(self.pm) +
            "\n\nThis is pv (variance of VI samples):\n" + str(self.pv) +
            "\n\nThis is qm (mean of true samples):\n" + str(self.qm) +
            "\n\nThis is qv (variance of true samples):\n" + str(self.qv) +
            "\n"
        )

        # 
        if outdir is not None:
            path = os.path.join(outdir, filename)
            with open(path, "w") as f:
                f.write(stats_str)
            print(f"Sample statistics saved to {path}")

        #print(stats_str)
        # 
        return self.pm, self.pv, self.qm, self.qv


"""Sourse: https://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py """
"""Sourse: https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py"""
    @staticmethod
    def gau_kl(pm, pv, qm, qv):
        """
        Kullback-Leibler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Diagonal covariances are assumed. Divergence is in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # 
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)
            + (iqv * pv).sum(axis)
            + (diff * iqv * diff).sum(axis)
            - len(pm)
        ))

    @staticmethod
    def _knn_distance(point, sample, k):
        norms = np.linalg.norm(sample - point, axis=1)
        return np.sort(norms)[k]

    @staticmethod
    def _verify_sample_shapes(s1, s2, k):
        assert len(s1.shape) == len(s2.shape) == 2
        assert s1.shape[1] == s2.shape[1]

    @staticmethod
    def naive_estimator(s1, s2, k):
        Runner._verify_sample_shapes(s1, s2, k)
        n, m = len(s1), len(s2)
        D = np.log(m / (n - 1))
        d = float(s1.shape[1])
        for p1 in s1:
            nu = Runner._knn_distance(p1, s2, k - 1)
            rho = Runner._knn_distance(p1, s1, k)
            D += (d / n) * np.log(nu / rho)
        return D

    @staticmethod
    def scipy_estimator(s1, s2, k):
        Runner._verify_sample_shapes(s1, s2, k)
        n, m = len(s1), len(s2)
        d = float(s1.shape[1])
        D = np.log(m / (n - 1))
        nu_d, nu_i = KDTree(s2).query(s1, k)
        rho_d, rhio_i = KDTree(s1).query(s1, k + 1)
        if k > 1:
            D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
        else:
            D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))
        return D

    @staticmethod
    def skl_estimator(s1, s2, k):
        Runner._verify_sample_shapes(s1, s2, k)
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
    def skl_efficient(s1, s2, k):
        Runner._verify_sample_shapes(s1, s2, k)
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
        return (d / n) * D + np.log(m / (n - 1))


    def kl_metrics(self, n_samples=10_000, k=10, outdir=None, filename="kl_metrics.txt", kl_div=None):
        pm, pv, qm, qv = self.print_statistics(n_samples=n_samples)
        s1 = self.get_vi_samples(n_samples)
        s2 = self.get_true_samples(n_samples)
        kl_div = self.gau_kl(pm, pv, qm, qv)
        naive = self.naive_estimator(s1, s2, k)
        scipy_val = self.scipy_estimator(s1, s2, k)
        skl_val = self.skl_estimator(s1, s2, k)
        skl_e_val = self.skl_efficient(s1, s2, k)

        out_path = os.path.join(outdir, filename)
        with open(out_path, "w") as f:
            f.write(f"Parametric KL: {kl_div:.8f}\n")
            f.write(f"KNN naive KL: {naive:.8f}\n")
            f.write(f"KNN scipy KL: {scipy_val:.8f}\n")
            f.write(f"KNN sklearn KL: {skl_val:.8f}\n")
            f.write(f"KNN sklearn fast KL: {skl_e_val:.8f}\n")
        #print(f"\nKL divergence between VI approximation and True Normal: {kl_div:.8f} nats")
        #return kl_div




    def print_summary(self):
        print("Runner Summary:")
        print(f"  dim: {self.dim}")
        #print(f"  mean: {self.mean}")
        #print(f"  cov:\n{self.cov}")
        #print(f"  prior_bounds: {self.prior_bounds}")
        print(f"  steps: {self.steps}")
        print(f"  batch_size: {self.batch_size}")

    




# run function:
if __name__ == "__main__":
    runner = Runner()
    flow, losses = runner.run()
    runner.print_summary()
    runner.plot_vi_vs_true()  
    #runner.print_kl_divergence()
    
    # saving results
    results_dir = runner.get_next_available_outdir(base_dir="results")
    runner.get_vi_samples(n_samples=10_000, outdir=results_dir)
    runner.get_true_samples(n_samples=10_000, outdir=results_dir)
    runner.plot_losses(outdir=results_dir)
    runner.plot_vi_vs_true(outdir=results_dir)
    runner.print_statistics(outdir=results_dir)
    runner.kl_metrics(outdir=results_dir)


