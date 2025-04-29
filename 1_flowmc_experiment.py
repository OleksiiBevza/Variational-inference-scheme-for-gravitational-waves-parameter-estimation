# this should be installed in conda environment 
# ! pip install flowMC
# ! pip install corner
# ! pip install numpy
# ! pip install matplotlib.pyplot
# ! pip install arviz


import jax
import jax.numpy as jnp
from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.strategy.optimization import AdamOptimization
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.Sampler import Sampler
import numpy as np

import jax
import jax.numpy as jnp  
import optax  
import equinox as eqx  

from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float, Array
from typing import Any

from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

import matplotlib.pyplot as plt
import corner
import arviz as az



class BaseSampler:
    """This class specifies all parameters and runs a sampler flowMC"""
    def __init__(
        self,
        n_dims,     # specify in distribution function
        n_chains,   # specify in distribution function
        target_fn,
        seed=42,
        n_local_steps=100,
        n_global_steps=10,
        n_training_loops=20,
        n_production_loops=20,
        n_epochs=5,
        rq_spline_hidden_units=[32, 32],
        rq_spline_n_bins=8,
        rq_spline_n_layers=4,
        learning_rate=5e-3,
        batch_size=10000,
        n_max_examples=10000,
        verbose=False,
    ):
        self.n_dims = n_dims
        self.n_chains = n_chains
        self.rng_key = jax.random.PRNGKey(seed)
        self.data = {"data": jnp.arange(n_dims).astype(jnp.float32)}
        self.target_fn = target_fn

        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.bundle = RQSpline_MALA_Bundle(
            subkey,
            n_chains,
            n_dims,
            self.target_fn,
            n_local_steps,
            n_global_steps,
            n_training_loops,
            n_production_loops,
            n_epochs,
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            verbose=verbose,
        )

        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        self.sampler = Sampler(
            n_dims,
            n_chains,
            self.rng_key,
            resource_strategy_bundles=self.bundle,
        )

    def run_sampling(self):
        self.sampler.sample(self.initial_position, self.data)


class DualMoonSampler(BaseSampler):
    """This class specifies a Duala Moon distribution function"""
    def __init__(self, n_dims=5, n_chains=20, seed=42, **kwargs):    # specify number of dimensions
        def target_dual_moon(x: jax.Array, data: dict) -> float:
            term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
            term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
            term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
            return -(term1 - logsumexp(term2) - logsumexp(term3))

        super().__init__(n_dims, n_chains, target_dual_moon, seed, **kwargs)


class GaussianSampler(BaseSampler):
    """This class specifies a Gaussian distribution function"""
    def __init__(self, n_dims=5, n_chains=20, seed=42, **kwargs):    # specify number of dimensions
        def target_normal(x, data):
            return -0.5 * jnp.sum(x**2)

        super().__init__(n_dims, n_chains, target_normal, seed, **kwargs)


class RosenbrockSampler(BaseSampler):
    """This class specifies a Rosenbrock distribution function"""
    def __init__(self, n_dims=5, n_chains=20, seed=42, **kwargs):    # specify number of dimensions
        def target_rosenbrock(x, data):
            return -jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        super().__init__(n_dims, n_chains, target_rosenbrock, seed, **kwargs)


class Experiment:
    """This class takes data from BaseSampler and provide results in the form of plots"""
    def __init__(self, sampler: BaseSampler):
        self.sampler = sampler

    def run(self):
        self.sampler.run_sampling()

    def plot_losses(self):
        loss_data = self.sampler.sampler.resources["loss_buffer"].data
        plt.figure(figsize=(5, 3))
        plt.plot(loss_data.reshape(-1, 1))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.show()

    def plot_corner(self):
        labels = [f"x{i}" for i in range(self.sampler.n_dims)]
        chains = self.sampler.sampler.resources["positions_production"].data
        nf_samples = self.sampler.sampler.resources["model"].sample(jax.random.PRNGKey(2046), 10000)

        fig = plt.figure(figsize=(6, 6))
        corner.corner(np.array(chains.reshape(-1, self.sampler.n_dims)), fig=fig, labels=labels)
        plt.suptitle("MCMC Samples")
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(6, 6))
        corner.corner(np.array(nf_samples), fig=fig, labels=labels)
        plt.suptitle("Normalizing Flow Samples")
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self):
        local_accs = self.sampler.sampler.resources["local_accs_production"].data
        global_accs = self.sampler.sampler.resources["global_accs_production"].data
        log_prob = self.sampler.sampler.resources["log_prob_production"].data

        mean_local_accs = np.mean(local_accs, axis=0)
        mean_global_accs = np.mean(global_accs, axis=0)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(mean_local_accs)
        ax[0].set_title("Local acceptance rate")
        ax[0].set_xlabel("Local steps")

        ax[1].plot(mean_global_accs)
        ax[1].set_title("Global acceptance rate")
        ax[1].set_xlabel("Global steps")

        ax[2].plot(log_prob[::5, ::20].T, lw=1, alpha=0.5)
        ax[2].set_title("Log probability")
        plt.tight_layout()
        plt.show()

    def plot_rhat(self, n_step_group=7):
        chains = np.array(self.sampler.sampler.resources["positions_production"].data)
        n_steps = chains.shape[1]
        n_group_step = n_steps // n_step_group

        rhat_s = np.array(
            [
                [
                    az.rhat(chains[:, : (i + 1) * n_group_step, j], method="rank")
                    for i in range(n_step_group)
                ]
                for j in range(self.sampler.n_dims)
            ]
        )
        iterations = np.linspace(0, n_steps, n_step_group)

        plt.figure(figsize=(5, 3))
        plt.plot(iterations, rhat_s.T, "-o", label=[f"x{i}" for i in range(self.sampler.n_dims)])
        plt.axhline(1, c="k", ls="--")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\hat{R}$")
        plt.title("R-hat Diagnostic")
        plt.legend()
        plt.tight_layout()
        plt.show()




# Those commands run experiment
sampler = RosenbrockSampler()    # choose class with distribution function you want to run
experiment = Experiment(sampler)
experiment.run()
experiment.plot_losses()
experiment.plot_corner()
experiment.plot_diagnostics()
experiment.plot_rhat()

