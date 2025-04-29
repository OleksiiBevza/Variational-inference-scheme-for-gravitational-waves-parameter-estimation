"""
Code for generating and running toy problems with flowMC
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import corner

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Any

from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.strategy.optimization import AdamOptimization
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.Sampler import Sampler

SUPPORTED_EXPERIMENTS = ["gaussian", "dualmoon", "rosenbrock"]

### The argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
parser.add_argument(
    "--experiment-type",
    choices=["gaussian", "dualmoon", "rosenbrock"],
    required=True,
    help="Which experiment to run."
)
parser.add_argument(
    "--n-dims",
    type=int,
    required=True,
    help="Number of dimensions."
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="The output directory, where things will be stored"
)

# Everything below here are hyperparameters for the flowMC algorithms. 
parser.add_argument(
    "--n-local-steps",
    type=int,
    default=20,
    help="Number of local steps."
)
parser.add_argument(
    "--n-global-steps",
    type=int,
    default=50,
    help="Number of global steps."
)
parser.add_argument(
    "--n-training-loops",
    type=int,
    default=20,
    help="Number of training loops."
)
parser.add_argument(
    "--mala-step-size",
    type=float,
    default=1e-4,
    help="Step size for the MALA proposal (local sampler)."
)
parser.add_argument(
    "--n-production-loops",
    type=int,
    default=20,
    help="Number of production loops."
)
parser.add_argument(
    "--n-epochs",
    type=int,
    default=5,
    help="Number of epochs to train the NF."
)
parser.add_argument(
    "--n-chains",
    type=int,
    default=20,
    help="Number of Markov chains to process in parallel."
)
parser.add_argument(
    "--rq-spline-hidden-units",
    default=32,
    help="Spline number of hidden units (used in the NF)."
)
parser.add_argument(
    "--rq-spline-n-bins",
    type=int,
    default=8,
    help="Number of spline bins used in the NF."
)
parser.add_argument(
    "--rq-spline-n-layers",
    type=int,
    default=4,
    help="Number of spline layers used in the NF."
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate for the NF training."
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=10000,
    help="Batch size for NF training."
)
parser.add_argument(
    "--n-max-examples",
    type=int,
    default=10000,
    help="Maximum number of examples for NF."
)
parser.add_argument(
    "--show-initial-positions",
    action="store_true",
    help="Show initial chain positions."
)


class FlowMCExperimentRunner:
    """
    Base class storing everything shared between different run experiments
    """
    def __init__(self, args):
        
        # Process the argparse args into params:
        self.params = vars(args)
        
        # Check the given outdirectory:
        if not os.path.exists(self.params["outdir"]):
            print(f"The output directory {self.params['outdir']} does not exist. Creating it now . . .")
            os.makedirs(self.params["outdir"])
        else:
            print(f"WARNING: The output directory {self.params['outdir']} already exists. This will overwrite any previous data in this directory.")
        
        # Check if experiment type is allowed/supported:
        if not self.params["experiment_type"] in SUPPORTED_EXPERIMENTS:
            raise ValueError(f"Experiment type {self.params['experiment_type']} is not supported. Supported types are: {SUPPORTED_EXPERIMENTS}")
        
        # Show the parameters to the screen/log file
        print(f"Passed parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")
            
        # Specify the desired target function based on the experiment type
        if self.params["experiment_type"] == "gaussian":
            # TODO: generalize the target (see below) and print the user specified information here
            print(f"Setting the target function to a Gaussian distribution.")
            self.target_fn = self.target_normal
        elif self.params["experiment_type"] == "dualmoon":
            print(f"Setting the target function to a dual moon distribution.")
            self.target_fn = self.target_dual_moon
        elif self.params["experiment_type"] == "rosenbrock":
            print(f"Setting the target function to a Rosenbrock distribution.")
            self.target_fn = self.target_rosenbrock

    def target_normal(self, x, data):
        # TODO: generalize to a general Gaussian distribution
        # TODO: generalize to a mixture of several of such Gaussians (and make these part of the argparse)
        return -0.5 * jnp.sum(x**2)

    def target_dual_moon(self, x: jnp.ndarray, data: dict[str, Any]) -> jnp.ndarray:
        term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
        term2 = -0.5 * ((x[0] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
        term3 = -0.5 * ((x[1] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
        return -(term1 - logsumexp(term2) - logsumexp(term3))

    def target_rosenbrock(self, x: jnp.ndarray, data: dict[str, Any]) -> jnp.ndarray:
        return -jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def run_experiment(self):
        """
        Run the sampler for the chosen experiment
        """
        dim = self.params["n_dims"]
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        
        # TODO: generalize the distribution from which the initial samples are generated
        initial_position = jax.random.normal(subkey, shape=(self.params["n_chains"], dim))

        if self.params["show_initial_positions"]:
            print("Initial chain positions were:")
            print(initial_position)

        # TODO: check if OK like this
        # data = {"data": jnp.arange(dim).astype(jnp.float32)}
        data = {} # this is unused in our targets and therefore not important

        # Pass the hyperparameters to flowMC
        bundle = RQSpline_MALA_Bundle(
            subkey,
            self.params["n_chains"],
            dim,
            self.target_fn,
            self.params["n_local_steps"],
            self.params["n_global_steps"],
            self.params["n_training_loops"],
            self.params["n_production_loops"],
            self.params["n_epochs"],
            mala_step_size=self.params["n_epochs"],
            rq_spline_hidden_units=[self.params["rq_spline_hidden_units"], self.params["rq_spline_hidden_units"]],
            rq_spline_n_bins=self.params["rq_spline_n_bins"],
            rq_spline_n_layers=self.params["rq_spline_n_layers"],
            learning_rate=self.params["learning_rate"],
            batch_size=self.params["batch_size"],
            n_max_examples=self.params["n_max_examples"],
            verbose=False
        )
        
        # Define the Sampler
        self.sampler = Sampler(
            dim,
            self.params["n_chains"],
            rng_key,
            resource_strategy_bundles=bundle,
        )
        
        # Start sampling:
        print(f"Starting the sampling now . . .")
        self.sampler.sample(initial_position, data)
        print(f"Sampling complete!")
        
        return self.sampler

    def plot_diagnostics(self):
        """
        Make a diagnosis plot. Note: this assumes that the sampler has been run and the data is available, so can only be called after the experiment has been run.
        """
        
        print(f"Making diagnosis plot . . .")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        local_accs = self.sampler.resources["local_accs_production"].data
        global_accs = self.sampler.resources["global_accs_production"].data
        log_prob = self.sampler.resources["log_prob_production"].data

        mean_local_accs = np.mean(local_accs, axis=0)
        mean_global_accs = np.mean(global_accs, axis=0)

        axes[0].plot(mean_local_accs)
        axes[0].set_title(f"Local Acceptance Rate")

        axes[1].plot(mean_global_accs)
        axes[1].set_title(f"Global Acceptance Rate")

        axes[2].plot(log_prob[::5, ::20].T, lw=1, alpha=0.5)
        axes[2].set_title(f"Log Probability")

        plt.tight_layout()
        save_name = os.path.join(self.params["outdir"], 'acceptance_and_logprob.pdf')
        print(f"Saving diagnosis plots to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight") # best way to save plot is as PDF with bbox_inches = tight
        plt.close(fig)
        
        print(f"Making diagnosis plot . . . DONE!")

    def plot_corner(self):
        chains = self.sampler.resources["positions_production"].data
        nf_samples = self.sampler.resources["model"].sample(jax.random.PRNGKey(2046), 10_000)
        labels = [f"x{i}" for i in range(self.params["n_dims"])]

        # Plot the corner plot for the chains
        fig1 = plt.figure(figsize=(6, 6))
        corner.corner(np.array(chains.reshape(-1, self.params["n_dims"])), fig=fig1, labels=labels)
        save_name = os.path.join(self.params["outdir"], 'chains_corner_plot.pdf')
        print(f"Saving diagnosis plots to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close(fig1)

        # Plot the samples from the flow
        fig2 = plt.figure(figsize=(6, 6))
        save_name = os.path.join(self.params["outdir"], 'flow_samples_corner_plot.pdf')
        corner.corner(np.array(nf_samples), fig=fig2, labels=labels)
        print(f"Saving diagnosis plots to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close(fig2)

    def print_data(self):
        local_accs = self.sampler.resources["local_accs_production"].data
        global_accs = self.sampler.resources["global_accs_production"].data
        log_prob = self.sampler.resources["log_prob_production"].data
        print("Local Accs:", local_accs)
        print("Global Accs:", global_accs)
        print("Log Prob:", log_prob)



def main():
    # Get the arguments passed over from the command line, and create the experiment runner
    args = parser.parse_args()
    runner = FlowMCExperimentRunner(args)
    
    # Run the experiment and do some postprocessing
    runner.run_experiment()
    runner.plot_diagnostics()
    runner.print_data()
    runner.plot_corner()
    
    
if __name__ == "__main__":
    main()
