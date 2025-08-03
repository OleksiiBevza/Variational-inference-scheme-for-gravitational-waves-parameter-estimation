Approximating complex probability densities via optimization using flow-based variational inference (VI) could potentially provide a solution to current computational challenges in various domains. This research project presents a thorough literature review on the conceptual fundamentals of VI, including its enhancements and limitations in high-dimensional spaces. A VI sampler using a Block Neural Autoregressive Flow is being applied to a set of toy problems to test the potential applicability of this approximation technique for addressing the [gravitational-wave data analysis challenges](https://arxiv.org/abs/2312.11103) of the [Einstein Telescope project](https://www.einsteintelescope-emr.eu/en/). It is believed that this preliminary approach should be taken before applying VI directly to gravitational-wave analysis, due to conceptual constraints identified in the literature review.

The first toy problem concerns a single-component multivariate normal distribution with different mean vectors (µ) and covariance matrices (Σ), ranging from 1 up to 15 dimensions.
The second toy problem concerns a two-component multivariate normal distribution with different means, weights, and covariances, currently explored in 1 to 4 dimensions. The complexity of these toy problems will be increased in the near future.

The inspection of the results includes:
(i) Visual inspection of the true and VI distributions.
(ii) Comparison of the means, variances, and standard deviations of the true and VI distributions.
(iii) Examination of the parametric and nonparametric KL divergences between the true and VI distributions.



To run the script on Snellius, run `sbatch submit_job.sh`
