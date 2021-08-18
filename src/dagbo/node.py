from torch import Tensor
from gpytorch.constraints.constraints import GreaterThan, Positive
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood, _GaussianLikelihoodBase
from gpytorch.means.mean import Mean
from gpytorch.means.zero_mean import ZeroMean
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.torch_priors import GammaPrior
from .parametric_mean import ParametricMean
from typing import List, Optional, Union

MIN_INFERRED_NOISE_LEVEL = 1e-4

class Node(ExactGP):
    """
    An ExactGP with a configurable mean.
    Mean can subclass gpytorch.means.Mean or ParametricMean.
    Default mean is ZeroMean.
    Uses GaussianLikelihood and MaternKernel(nu=5/2).
    Priors for Likelihood and Kernel are borrowed from BoTorch SingleTaskGP
    """
    def __init__(self, input_names: List[str], output_name: str,
                 train_inputs: Tensor, train_targets: Tensor, 
                 mean: Optional[Union[Mean, ParametricMean]] = None,
                 covar: Optional[Kernel] = None,
                 likelihood: Optional[_GaussianLikelihoodBase] = None):
        """
        Args:
            input_names: a d-length List of the names of each input
                defining the order of the inputs in the innermost
                dimension of train_inputs.
            output_name: the names of the metric this node is modelling
            train_inputs: A batch_shape*q*d-dim Tensor of the training inputs
                The innermost dim, dim=-1, must follow the same order as
                input_names as this is how the data is supplied to the 
                parametric mean the DAG.
            train_targets: A batch_shape*q-dim Tensor of the training targets
        """
        batch_shape = train_inputs.shape[:-2]
        num_inputs = train_inputs.shape[-1]

        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            # Use GreaterThan constraint for Torch / SciPy fitting
            # Use Positive constraint for MCMC fitting (because MCMC does not play nicely
            # with constraints within the range of the prior)
            noise_constraint = GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior_mode,
            )
            # noise_constraint = Positive(initial_value=noise_prior_mode)
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_prior=noise_prior,
                noise_constraint=noise_constraint
            )
        super().__init__(train_inputs, train_targets, likelihood)

        self.input_names = input_names
        self.output_name = output_name

        self.mean = ConstantMean(batch_shape=batch_shape) if mean is None else mean

        # Use UniformPrior for MCMC, use GammaPrior for Torch/Scipy
        # loss = inf when using UniformPrior with Torch/Scipy so they can't learn
        # MCMC occasionally samples a very small value for lengthscale, which produces NaNs 
        #   and there is no way (afaik) to prevent MCMC from picking these very small values,
        #   even constraints don't work because MCMC samples from lengthscale as opposed to
        #   raw_lengthscale, so it simply throws an error if it samples an illegal lengthscale value
        self.covar = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=num_inputs,
                batch_shape=batch_shape,
                # lengthscale_prior=UniformPrior(0.01, 0.02)
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15)
        ) if covar is None else covar

        self.to(train_inputs)

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Returns the posterior of the latent Gaussian process at `x`
        when conditioned on the training data.

        Args:
            x: batch_shape*q*d-dim tensor
        """
        if isinstance(self.mean, ParametricMean):
            mean_x = self.mean(x, self.input_names)
        else:
            mean_x = self.mean(x)
        covar_x = self.covar(x)
        return MultivariateNormal(mean_x, covar_x)
