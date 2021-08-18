from torch import Tensor, Size
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.settings import fast_computations
from botorch.posteriors.gpytorch import GPyTorchPosterior
from typing import Optional, Type, TypeVar

T = TypeVar('T', bound='SampleAveragePosterior')
class SampleAveragePosterior(GPyTorchPosterior):
    """
    The use case for SampleAveragePosterior is for a model with some non-determinism 
    in the prediction step, for example, if the model is doing sampling internally.

    SampleAveragePosterior computes multiple posteriors for the same input points. This
    creates a Monte-Carlo approximation to model's non-determinism. Each one of these
    multiple posteriors is a Gaussian distribution, so the Monte-Carlo approximation 
    is actually a Gaussian mixture distribution. Since GPyTorchPosteriors have to be 
    normally distributed, SampleAveragePosterior then approximates this Gaussian 
    mixture distribution as a Gaussian distribution.

    There is no guarantee that this is a good approximation. However, if it is bad
    approximation then it probably means you shouldn't be using Gaussian processes
    for your problem anyway!
    
    SampleAveragePosterior hides all of this complexity. Use it by simply wrapping an
    existing GPyTorchPosterior in a SampleAveragePosterior. The outermost dim is then
    assumed to be the sample batch.

    # e.g.
    num_samples = 100
    original_shape = Size([10,5])
    # create multiple posteriors at identical input points
    expanded_test_X = test_X.unsqueeze(dim=0).expand(num_samples, *original_shape)
    # posterior.event_shape = Size([100,10,5])
    posterior = model.posterior(expanded_test_X)
    # wrap in SampleAveragePosterior, now posterior.event_shape = Size([10,5])
    posterior = SampleAveragePosterior.from_gpytorch_posterior(posterior)
    """

    def __init__(self, mvn: MultivariateNormal) -> None:
        """
        Args:
            mvn: Batch MultivariateNormal
                Outermost dimension (dim=0) of mvn is filled with samples.
                These samples will be combined by taking their mean.
        """
        super().__init__(mvn)

    @property
    def num_samples(self) -> int:
        """The number of samples that the posterior is averaged over."""
        return super().event_shape[0]

    @property
    def event_shape(self) -> Size():
        """The event shape (i.e. the shape of a single sample) of the posterior."""
        return super().event_shape[1:]

    @property
    def mean(self) -> Tensor:
        """
        The posterior mean.
        = mean of each Gaussian

        Reference:
        https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        We have num_samples Gaussians, and since each Gaussian is drawn using Monte Carlo, then each Gaussian has equal weight
        """
        return super().mean.mean(dim=0)

    @property
    def variance(self) -> Tensor:
        """
        The posterior variance.
        = avg of variance + gvg of squared mean - square of avg mean

        Reference:
        https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        We have num_samples Gaussians, and since each Gaussian is drawn using Monte Carlo, then each Gaussian has equal weight
        """
        return super().variance.mean(dim=0) + super().mean.square().mean(dim=0) - super().mean.mean(dim=0).square()

    def _replacement_super_rsample(
        self,
        sample_shape: Optional[Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample from the posterior (with gradients).

        This is basically a copy of super().rsample(), however there
        is one line in super().rsample() where the base samples are reshaped
        using self.event_shape and this fails because I have re-written event shape.
        Effectively I have two event shapes, an internal event shape which is
        Size([self.num_samples, *self.event_shape]), and an external event shape
        which is self.event_shape.
        
        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event_shape`-dim Tensor of samples from the posterior.
        """
        if sample_shape is None:
            sample_shape = Size([1])
        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")
            # get base_samples to the correct shape
            # THIS IS THE LINE I HAVE CHANGED
            base_samples = base_samples.expand(sample_shape + Size([self.num_samples]) + self.event_shape)
            # remove output dimension in single output case
            if not self._is_mt:
                base_samples = base_samples.squeeze(-1)
        with fast_computations(covar_root_decomposition=False):
            samples = self.mvn.rsample(
                sample_shape=sample_shape, base_samples=base_samples
            )
        # make sure there always is an output dimension
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    def rsample(self, sample_shape: Optional[Size] = None, base_samples: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            sample_shape: shape of the additional sample dimensions
            base_samples: sample_shape * event_shape Tensor
        """
        sample_average_dim = len(sample_shape)
        if base_samples is not None:
            # expand
            base_samples = base_samples.unsqueeze(sample_average_dim).expand(*sample_shape, self.num_samples, *self.event_shape)
        samples = self._replacement_super_rsample(sample_shape=sample_shape, base_samples=base_samples)
        # compact
        return samples.mean(dim=sample_average_dim)

    @classmethod
    def from_gpytorch_posterior(cls: Type[T], posterior: GPyTorchPosterior) -> T:
        return cls(mvn=posterior.mvn)

