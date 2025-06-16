import jax.numpy as jnp


class VariationalInference:




########################################################################################################################################################
# PACKAGES
########################################################################################################################################################
    # packages for variational inference and flow
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox

    from flowjax.bijections import (
        Affine as AffinePositiveScale,
        Chain,
        Exp,
        Identity,
        Stack,
        Tanh,
    )
    from flowjax.distributions import StandardNormal, Transformed, Uniform
    from flowjax.flows import block_neural_autoregressive_flow
    from paramax.wrappers import non_trainable, NonTrainable

    import time
    import jax_tqdm
    import optax


# packages for diagnostics
    import optax
    import jax
    from jax import random
    import corner 
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.stats import multivariate_normal


# packages for parsing
    #import os
    import argparse
    import arviz as az
    import xarray as xr
    import re
    import json









########################################################################################################################################################
# SAVING THE RESULTS
######################################################################################################################################################## 
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
        import os
        import re

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









########################################################################################################################################################
# DISTRIBUTION EXPERIMENT
########################################################################################################################################################     
# -- Multivariate Normal Likelihood as a Static Method/Nested Class --
    @staticmethod
    def MultivariateNormalLikelihood(dim, mean, cov):
        class _MultivariateNormalLikelihood:
            
            def __init__(self, dim, mean, cov):
                """
                Multivariate normal likelihood for experiment with different mean and covariance.

                Args:
                    dim: Dimension of the distribution
                    mean: Mean vector
                    cov: Full covariance matrix
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
                variance = 1.8 * jnp.trace(self.cov)
                return log_likelihood, variance

            def _log_prob(self, x):
                diff = x - self.mean
                cov_inv = jnp.linalg.inv(self.cov)
                log_det_cov = jnp.linalg.slogdet(self.cov)[1]
                quad_form = diff @ cov_inv @ diff
                return -0.5 * (self.dim * jnp.log(2 * jnp.pi) + log_det_cov + quad_form)

            def sample(self, size=None):
                import numpy as np
                return np.random.multivariate_normal(self.mean, self.cov, size=size)
        return _MultivariateNormalLikelihood(dim, mean, cov)
    










########################################################################################################################################################
# FLOW
########################################################################################################################################################
    @staticmethod
    def Affine(loc = 0, scale = 1):
        affine = VariationalInference.AffinePositiveScale(loc, scale)
        loc, scale = VariationalInference.jnp.broadcast_arrays(
            affine.loc, VariationalInference.jnp.asarray(scale, dtype = float),
        )
        affine = VariationalInference.equinox.tree_at(lambda tree: tree.scale, affine, scale)
        return affine

    @staticmethod
    def Logistic(shape = ()):
        loc = VariationalInference.jnp.ones(shape) * 0.5
        scale = VariationalInference.jnp.ones(shape) * 0.5
        return VariationalInference.Chain([VariationalInference.Tanh(shape), VariationalInference.Affine(loc, scale)])

    @staticmethod
    def UnivariateBounder(bounds = None):
        if (bounds is None) or all(bound is None for bound in bounds):
            return VariationalInference.Identity()
        elif any(bound is None for bound in bounds):
            if bounds[0] is None:
                loc = bounds[1]
                scale = -1
            elif bounds[1] is None:
                loc = bounds[0]
                scale = 1
            return VariationalInference.Chain([VariationalInference.Exp(), VariationalInference.Affine(loc, scale)])
        else:
            loc = bounds[0]
            scale = bounds[1] - bounds[0]
            return VariationalInference.Chain([VariationalInference.Logistic(), VariationalInference.Affine(loc, scale)])

    @staticmethod
    def Bounder(bounds):
        return VariationalInference.Stack(list(map(VariationalInference.UnivariateBounder, bounds)))

    @staticmethod
    def bound_from_unbound(flow, bounds = None):
        bounder = VariationalInference.Bounder(bounds)
        if all(type(b) is VariationalInference.Identity for b in bounder.bijections):
            bijection = flow.bijection
        else:
            bijection = VariationalInference.Chain([flow.bijection, VariationalInference.non_trainable(bounder)])
        return VariationalInference.Transformed(VariationalInference.non_trainable(flow.base_dist), bijection)

    @staticmethod
    def default_flow(key, bounds, **kwargs):
        """
        Here parameters related to flow are defined directly
        nn_depth
        nn_block_dim
        flow_layers
        """
        default_kwargs = dict(
            key = key,
            base_dist = VariationalInference.StandardNormal(shape = (len(bounds),)),
            invert = False,
            nn_depth = 1,           # by default was given 1
            nn_block_dim = 16,      # by default was given 8
            flow_layers = 1,        # by default was given 1
        )
        for arg in kwargs:
            default_kwargs[arg] = kwargs[arg]
        flow = VariationalInference.block_neural_autoregressive_flow(**default_kwargs)
        return VariationalInference.bound_from_unbound(flow, bounds)













########################################################################################################################################################
# VARIATIONAL INFERENCE SAMPLER
########################################################################################################################################################
    @staticmethod
    def get_prior(bounds):
        lo = VariationalInference.jnp.array(bounds)[:, 0]
        hi = VariationalInference.jnp.array(bounds)[:, 1]
        return VariationalInference.Uniform(minval = lo, maxval = hi)

    @staticmethod
    def get_log_likelihood(likelihood = None, return_variance = False):
        if likelihood is None:
            if return_variance:
                return lambda parameters: (0.0, 0.0)
            return lambda parameters: 0.0   # by default was given 0.0

        if return_variance:
            def log_likelihood_and_variance(parameters):
                likelihood.parameters.update(parameters)
                return likelihood.ln_likelihood_and_variance()
            return log_likelihood_and_variance

        def log_likelihood(parameters):
            likelihood.parameters.update(parameters)
            return likelihood.log_likelihood_ratio()
        return log_likelihood

    @staticmethod
    def likelihood_extras(likelihood, parameters):
        likelihood.parameters.update(parameters)
        likelihood.parameters, added_keys = likelihood.conversion_function(
            likelihood.parameters,
        )
        likelihood.hyper_prior.parameters.update(parameters)

        log_bayes_factors, variances = \
            likelihood._compute_per_event_ln_bayes_factors()

        detection_efficiency, detection_variance = \
            likelihood.selection_function.detection_efficiency(parameters)

        selection = - likelihood.n_posteriors * VariationalInference.jnp.log(detection_efficiency)
        selection_variance = (
            likelihood.n_posteriors ** 2
            * detection_variance
            / detection_efficiency ** 2
        )

        log_likelihood = VariationalInference.jnp.sum(log_bayes_factors) + selection
        variance = VariationalInference.jnp.sum(variances) + selection_variance

        return dict(
            log_likelihood = log_likelihood,
            variance = variance,
            log_bayes_factors = log_bayes_factors,
            variances = variances,
            detection_efficiency = detection_efficiency,
            detection_variance = detection_variance,
            selection = selection,
            selection_variance = selection_variance,
        )

    @staticmethod
    def trainer(
        key,
        prior_bounds = None,
        likelihood = None,
        vmap = True,
        flow = None,
        batch_size = None,
        steps = None,
        learning_rate = None,
        optimizer = None,
        taper = None,
        temper_schedule = None,
        **tqdm_kwargs,
    ):
        print('GWAX - getting ready...')

        names = tuple(prior_bounds.keys())
        bounds = tuple(prior_bounds.values())
        prior = VariationalInference.get_prior(bounds)

        _log_likelihood_and_variance = VariationalInference.get_log_likelihood(likelihood, True)
        if vmap:
            log_likelihood_and_variance = VariationalInference.jax.vmap(_log_likelihood_and_variance)
        else:
            log_likelihood_and_variance = lambda parameters: VariationalInference.jax.lax.map(
                _log_likelihood_and_variance, parameters,
            )

        if taper is None:
            taper = lambda variance: 0.0

        def log_target(samples):
            parameters = dict(zip(names, samples.T))
            log_lkls, variances = log_likelihood_and_variance(parameters)
            return prior.log_prob(samples) + log_lkls + taper(variances)

        if flow is None:
            key, _key = VariationalInference.jax.random.split(key)
            flow = VariationalInference.default_flow(_key, bounds)

        params, static = VariationalInference.equinox.partition(
            pytree = flow,
            filter_spec = VariationalInference.equinox.is_inexact_array,
            is_leaf = lambda leaf: isinstance(leaf, VariationalInference.NonTrainable),
        )

        def loss_fn(params, key, step):
            flow = VariationalInference.equinox.combine(params, static)
            samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
            log_targets = log_target(samples) * temper_schedule(step)
            return VariationalInference.jnp.mean(log_flows - log_targets)

        if optimizer is None:
            optimizer = VariationalInference.optax.adam
        if callable(optimizer):
            optimizer = optimizer(learning_rate)

        state = optimizer.init(params)

        if temper_schedule is None:
            temper_schedule = lambda step: 1.0

        tqdm_defaults = dict(
            print_rate = 1,
            tqdm_type = 'auto',
            desc = 'GWAX - variational training',
        )
        for arg in tqdm_kwargs:
            tqdm_defaults[arg] = tqdm_kwargs[arg]

        @VariationalInference.jax_tqdm.scan_tqdm(steps, **tqdm_defaults)
        @VariationalInference.equinox.filter_jit
        def update(carry, step):
            key, params, state = carry
            key, _key = VariationalInference.jax.random.split(key)
            loss, grad = VariationalInference.equinox.filter_value_and_grad(loss_fn)(params, _key, step)
            updates, state = optimizer.update(grad, state, params)
            params = VariationalInference.equinox.apply_updates(params, updates)
            return (key, params, state), loss

        print('GWAX - JAX jitting...')
        t0 = VariationalInference.time.time()
        (key, params, state), losses = VariationalInference.jax.lax.scan(
            update, (key, params, state), VariationalInference.jnp.arange(steps),
        )
        flow = VariationalInference.equinox.combine(params, static)
        print(f'GWAX: Total time = {VariationalInference.time.time() - t0} s')

        return flow, losses

    @staticmethod
    def _importance(log_weights, n = None):
        if n is None:
            n = log_weights.size
        log_evidence = VariationalInference.jax.nn.logsumexp(log_weights) - VariationalInference.jnp.log(n)
        log_sq_mean = 2 * log_evidence
        log_mean_sq = VariationalInference.jax.nn.logsumexp(2 * log_weights) - VariationalInference.jnp.log(n)
        efficiency = VariationalInference.jnp.exp(log_sq_mean - log_mean_sq)
        ess = efficiency * n
        log_evidence_variance = 1 / ess - 1 / n
        log_evidence_sigma = log_evidence_variance ** 0.5
        return dict(
            efficiency = efficiency,
            log_evidence = log_evidence,
            log_evidence_sigma = log_evidence_sigma,
        )

    @staticmethod
    def importance(
        key,
        prior_bounds,
        likelihood = None,
        flow = None,
        n = 10_000,
        loop = 'scan',   # 'vmap', 'map', 'scan', or 'for'
        **tqdm_kwargs,
    ):
        _log_likelihood = VariationalInference.get_log_likelihood(likelihood, False)
        _log_likelihood = VariationalInference.equinox.filter_jit(_log_likelihood)

        loop = loop.lower()
        if loop == 'vmap':
            log_likelihood = VariationalInference.jax.vmap(_log_likelihood)
        elif loop == 'map':
            log_likelihood = lambda parameters: VariationalInference.jax.lax.map(
                _log_likelihood, parameters,
            )
        elif loop == 'scan':
            tqdm_defaults = dict(
                print_rate = 1,
                tqdm_type = 'auto',
                desc = 'GWAX - importance sampling',
            )
            for arg in tqdm_kwargs:
                tqdm_defaults[arg] = tqdm_kwargs[arg]
            log_likelihood = lambda parameters: VariationalInference.jax.lax.scan(
                VariationalInference.jax_tqdm.scan_tqdm(n, **tqdm_defaults)(
                    lambda carry, ip: (None, _log_likelihood(ip[1])),
                ),
                None,
                (VariationalInference.jnp.arange(n), parameters),
            )[1]
        else:
            raise ValueError(
                'loop must be \'vmap\', \'map\', or \'scan\' (default \'scan\') '
                f'but got \'{loop}\'',
            )

        names = tuple(prior_bounds.keys())
        bounds = tuple(prior_bounds.values())
        prior = VariationalInference.get_prior(bounds)
        flow = prior if flow is None else flow

        samples, log_flows = flow.sample_and_log_prob(key, (n,))
        log_priors = prior.log_prob(samples)
        parameters = dict(zip(names, samples.T))
        log_lkls = log_likelihood(parameters)
        log_weights = log_priors + log_lkls - log_flows

        return dict(
            samples = samples,
            log_weights = log_weights,
            **VariationalInference._importance(log_weights),
        )
    


 






########################################################################################################################################################
# EXPERIMENT PARAMETERS (HERE ALMOST ALL PARAMETERS FOR EXPERIMENT AERE DEFINED)
######################################################################################################################################################## 
 
    
    @staticmethod
    def experiment_parameters(seed=0):
        """
        Return experiment parameters for a n-dim Gaussian VI problem.
        Almost all the parameters that needed to be passed are defined

        Args:
            seed: for reproducibility of covariance matrix, default: 0

        Returns:
            dict with keys:
                dim, key, mean, cov, likelihood, batch_size, prior_bounds1
        """
        import numpy as np
        import jax
        from jax import random

        # number of dimensions
        dim = 15
        key = random.key(0)

        # different means for different dimensions
        mean = np.array([
            -5.0, -3.0, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0,
            3.5, 3.7, 4.0, 4.5, 4.8, 5.0
        ])

        # covariance for multidimensional problem
        np.random.seed(seed)
        A = np.random.uniform(-1.0, 1.0, dim**2).reshape((dim, dim))
        cov = np.dot(A, A.transpose())

        likelihood = VariationalInference.MultivariateNormalLikelihood(dim, mean, cov)

        batch_size = 10000

        # prior bounds for every dimension
        prior_bounds1 = dict(
            x0 = [-9.5, -0.5],
            x1 = [-7.5, 1.5],
            x2 = [-5.5, 3.5],
            x3 = [-4.5, 4.5],
            x4 = [-3.5, 5.5],
            x5 = [-3.0, 6.0],
            x6 = [-2.5, 6.5],
            x7 = [-2.0, 7.0],
            x8 = [-1.5, 7.5],
            x9 = [-1.0, 8.0],
            x10 = [-0.8, 8.2],
            x11 = [-0.5, 8.5],
            x12 = [0.0, 9.0],
            x13 = [0.3, 9.3],
            x14 = [0.5, 9.5],
        )

        return dict(
            dim=dim,
            key=key,
            mean=mean,
            cov=cov,
            likelihood=likelihood,
            batch_size=batch_size,
            prior_bounds1=prior_bounds1,
        )


    @staticmethod
    def optimization_parameters(steps=1000, base_lr=1e-3):
        """
        Returns optimization parameters: steps, learning rate, optimizer.

        Args:
            steps: number of optimization steps: 1000 by default
            base_lr: base learning rate is  1e-3

        Returns:
            dict with keys:
                steps, learning_rate, optimizer
        """
        import optax

        learning_rate = optax.cosine_decay_schedule(base_lr, steps)
        optimizer = optax.adam(learning_rate)

        return dict(
            steps=steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )










########################################################################################################################################################
# DIAGNOSTICS
######################################################################################################################################################## 


    # function extracts a specified number of samples from dictionary.
    # these samples are also stored in json file in results folder
    @staticmethod
    def vi_samples(flow, outdir, n_samples, seed=1):
        import jax
        import numpy as np
        import json
        import os
        samples = np.array(flow.sample(jax.random.key(seed), (n_samples,)))

        # save in JSON in results
        samples_list = samples.tolist()
        out_path = os.path.join(outdir, "vi_samples.json")
        with open(out_path, "w") as f:
            json.dump(samples_list, f)
        print(f"VI samples saved to {out_path}")
        return samples
    


    # function creates a specified number of true samples for evaluation purposes VI vs true.
    # these samples are also stored in json file in results folder
    @staticmethod
    def true_distribution(likelihood, outdir, n_samples, seed=1):
        from scipy.stats import multivariate_normal
        import numpy as np
        import json
        import os

        mean1 = np.array(likelihood.mean)
        cov1 = np.array(likelihood.cov)
        true_dist = multivariate_normal(mean=mean1, cov=cov1, allow_singular=True)
        true_samples = true_dist.rvs(size=n_samples, random_state=seed)

        # save in JSON in results
        samples_list = true_samples.tolist()
        out_path = os.path.join(outdir, "true_samples.json")
        with open(out_path, "w") as f:
            json.dump(samples_list, f)
        print(f"True samples saved to {out_path}")
        return true_samples

    # this function plots loss function
    @staticmethod
    def plot_losses(losses, outdir=None):
        import matplotlib.pyplot as plt
        import os
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.axhline(1.7, c='r')
        plt.xlim(0, 2000)
        plt.ylim(0, 1000)
        plt.xlabel("Training step")
        plt.ylabel("VI loss")
        plt.title("VI Training Loss Curve")
        # save results
        if outdir is not None:
            plot_path = os.path.join(outdir, "vi_loss_curve.png")
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"VI loss curve plot saved to {plot_path}")
        #plt.show()
        plt.close()


    # this function plot corner plot
    @staticmethod
    def plot_corner(vi_samples, true_samples, outdir=None):
        import corner
        import matplotlib.pyplot as plt
        import os

        # plot VI samples
        hist_kwargs_vi = {"color": "blue", "density": True, "alpha": 0.5}
        fig = corner.corner(vi_samples, color="blue", label="VI Approximation",
                            hist_kwargs=hist_kwargs_vi, show_titles=True)

        # plot true samples
        hist_kwargs_true = {"color": "red", "density": True, "alpha": 0.5}
        corner.corner(true_samples, fig=fig, color="red", label="True Normal",
                    hist_kwargs=hist_kwargs_true, show_titles=True)

        handles = [plt.Line2D([], [], color="blue", label="VI Approximation"),
                    plt.Line2D([], [], color="red", label="True Normal")]
        fig.legend(handles=handles, loc="upper right")

        # save to results
        if outdir is not None:
            plot_path = os.path.join(outdir, "vi_vs_true_corner.png")
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Corner plot saved to {plot_path}")
        # plt.show()  
        plt.close(fig)  

    
    # this function prints mean and variance of the samples
    @staticmethod
    def print_sample_statistics(vi_samples, true_samples, outdir=None, filename="sample_statistics.txt"):
        """
        pm = mean of VI samples per dimension 
        pv = variance of VI samples per dimension
        qm = mean of true samples per dimension
        qv = variance of true samples per dimension
        """
        import numpy as np
        import os

        # compute means and variances
        pm = vi_samples.mean(axis=0)
        pv = vi_samples.var(axis=0)
        qm = true_samples.mean(axis=0)
        qv = true_samples.var(axis=0)

        np.set_printoptions(precision=4, suppress=True)

        stats_str = (
            "This is pm (mean of VI samples):\n" + str(pm) +
            "\n\nThis is pv (variance of VI samples):\n" + str(pv) +
            "\n\nThis is qm (mean of true samples):\n" + str(qm) +
            "\n\nThis is qv (variance of true samples):\n" + str(qv) +
            "\n"
        )

        # save to results
        if outdir is not None:
            path = os.path.join(outdir, filename)
            with open(path, "w") as f:
                f.write(stats_str)
            print(f"Sample statistics saved to {path}")

        #print(stats_str)


    @staticmethod
    def gau_kl(pm, pv, qm, qv):
        """
        Jensen-Shannon divergence between two Gaussians.  Also computes JS
        divergence between a single Gaussian pm,pv and a set of Gaussians
        qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        The result is bounded [0, 1]. If near 0, then distributions are similar, if near 1, dissimilar
        https://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
        """
        import numpy as np
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1./qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 *
                (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
                + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
                + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
                - len(pm)))                     # - N



    @staticmethod
    def knn_kl_estimators(vi_samples, true_samples, k=5):
        """
        KL-Divergence estimation through K-Nearest Neighbours

        This module provides four implementations of the K-NN divergence estimator of
        Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú.
        "Divergence estimation for multidimensional densities via
        k-nearest-neighbor distances." Information Theory, IEEE Transactions on
        55.5 (2009): 2392-2405.

        The implementations are through:
        numpy (naive_estimator)
        scipy (scipy_estimator)
        scikit-learn (skl_estimator / skl_efficient)

        No guarantees are made w.r.t the efficiency of these implementations.

        https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py

        """
        import numpy as np
        import warnings
        from scipy.spatial import KDTree
        from sklearn.neighbors import NearestNeighbors

        def knn_distance(point, sample, k):
            """
            Euclidean distance from `point` to it's `k`-Nearest
            Neighbour in `sample`
            This function works for points in arbitrary dimensional spaces.
            """
            # Compute all euclidean distances
            norms = np.linalg.norm(sample - point, axis=1)
            # Return the k-th nearest
            return np.sort(norms)[k]

        def verify_sample_shapes(s1, s2, k):
            # Expects [N, D]
            assert len(s1.shape) == len(s2.shape) == 2
            # Check dimensionality of sample is identical
            assert s1.shape[1] == s2.shape[1]

        def naive_estimator(s1, s2, k):
            """
            KL-Divergence estimator using brute-force (numpy) k-NN
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)
            """
            verify_sample_shapes(s1, s2, k)
            n, m = len(s1), len(s2)
            D = np.log(m / (n - 1))
            d = float(s1.shape[1])

            for p1 in s1:
                nu = knn_distance(p1, s2, k - 1)  # -1 because 'p1' is not in 's2'
                rho = knn_distance(p1, s1, k)
                D += (d / n) * np.log(nu / rho)
            return D

        def scipy_estimator(s1, s2, k):
            """
            KL-Divergence estimator using scipy's KDTree
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)
            """
            verify_sample_shapes(s1, s2, k)
            n, m = len(s1), len(s2)
            d = float(s1.shape[1])
            D = np.log(m / (n - 1))

            nu_d, _ = KDTree(s2).query(s1, k)
            rho_d, _ = KDTree(s1).query(s1, k + 1)

            # KTree.query returns different shape in k==1 vs k > 1
            if k > 1:
                D += (d / n) * np.sum(np.log(nu_d[:, -1] / rho_d[:, -1]))
            else:
                D += (d / n) * np.sum(np.log(nu_d / rho_d[:, -1]))
            return D

        def skl_estimator(s1, s2, k):
            """
            KL-Divergence estimator using scikit-learn's NearestNeighbours
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)
            """
            verify_sample_shapes(s1, s2, k)
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

        def skl_efficient(s1, s2, k):
            """
            An efficient version of the scikit-learn estimator by @LoryPack
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)

            Contributed by Lorenzo Pacchiardi (@LoryPack)
            """
            verify_sample_shapes(s1, s2, k)
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

        # run knn 
        naive = naive_estimator(vi_samples, true_samples, k=k)
        scipy_ = scipy_estimator(vi_samples, true_samples, k=k)
        skl = skl_estimator(vi_samples, true_samples, k=k)
        skl_eff = skl_efficient(vi_samples, true_samples, k=k)
        print("\nKNN-based KL Divergence Estimators (k={}):".format(k))
        print(f"  Naive estimator      : {naive:.4f}")
        print(f"  Scipy KDTree         : {scipy_:.4f}")
        print(f"  Scikit-learn (slow)  : {skl:.4f}")
        print(f"  Scikit-learn (fast)  : {skl_eff:.4f}")
        return [naive, scipy_, skl, skl_eff]


    # this function saves results
    @staticmethod
    def save_kl_results_to_txt(outdir, kl_analytical, knn_kl_list, filename="kl_metrics.txt"):
        import os
        out_path = os.path.join(outdir, filename)
        with open(out_path, "w") as f:
            f.write(f"Analytical KL: {kl_analytical:.8f}\n")
            f.write(f"KNN naive KL: {knn_kl_list[0]:.8f}\n")
            f.write(f"KNN scipy KL: {knn_kl_list[1]:.8f}\n")
            f.write(f"KNN sklearn KL: {knn_kl_list[2]:.8f}\n")
            f.write(f"KNN sklearn fast KL: {knn_kl_list[3]:.8f}\n")
        print(f"KL metrics saved to: {out_path}")


########################################################################################################################################################
# EXPERIMENT RUNNER
######################################################################################################################################################## 
    

    @staticmethod
    def run():
        params = VariationalInference.experiment_parameters(seed=0)
        optim_params = VariationalInference.optimization_parameters(steps=1000, base_lr=1e-3)

        key = params["key"]
        prior_bounds1 = params["prior_bounds1"]
        likelihood = params["likelihood"]
        batch_size = params["batch_size"]
        steps = optim_params["steps"]
        learning_rate = optim_params["learning_rate"]
        optimizer = optim_params["optimizer"]
       

        flow, losses = VariationalInference.trainer(
            key=key,
            prior_bounds=prior_bounds1,
            likelihood=likelihood,
            batch_size=batch_size,
            steps=steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            # flow = flow_parameters,
            vmap=True
        )
        print("Training finished.")
        print("Final loss:", losses[-1])

        # Create output directory ONCE
        outdir = VariationalInference.get_next_available_outdir("results", prefix="vi_results")


        # Get VI and true samples AND SAVE THEM!
        vi_samples = VariationalInference.vi_samples(flow, outdir, n_samples=10_000, seed=1)
        true_samples = VariationalInference.true_distribution(likelihood, outdir, n_samples=10_000, seed=1)

        # Now do your plots and statistics with the saved samples
        # Plot the losses
        VariationalInference.plot_losses(losses, outdir)
        VariationalInference.plot_corner(vi_samples, true_samples, outdir)
        VariationalInference.print_sample_statistics(vi_samples, true_samples, outdir)
        

        # Compute and print KL divergence
        pm = vi_samples.mean(axis=0)
        pv = vi_samples.var(axis=0)
        qm = true_samples.mean(axis=0)
        qv = true_samples.var(axis=0)

        kl_div = VariationalInference.gau_kl(pm, pv, qm, qv)
        knn_kl_list = VariationalInference.knn_kl_estimators(vi_samples, true_samples, k=5)
        VariationalInference.save_kl_results_to_txt(outdir, kl_div, knn_kl_list)


        # print(f"\nKL divergence between VI approximation and True Normal: {kl_div:.4f} nats")


        return vi_samples, true_samples
















# run experiment
if __name__ == "__main__":
    VariationalInference.run()