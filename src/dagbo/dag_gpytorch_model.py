from torch import Tensor
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from typing import Any, List, Union

from .sample_average_posterior import SampleAveragePosterior

class DagGPyTorchModel(GPyTorchModel):
    num_samples: int

    def posterior(self, X: Tensor, observation_noise: Union[bool, Tensor] = False, **kwargs: Any) -> GPyTorchPosterior:
        """Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        self.eval()  # make sure model is in eval mode

        original_shape = X.shape
        # create multiple posteriors at identical input points
        expanded_X = X.unsqueeze(dim=0).expand(self.num_samples, *original_shape)

        with gpt_posterior_settings():
            mvn = self(expanded_X)
            if observation_noise is not False:
                raise NotImplementedError("Observation noise is not yet supported for DagGPyTorch models.")
        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        posterior = SampleAveragePosterior.from_gpytorch_posterior(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        raise NotImplementedError("Condition on observations is not yet supported for DagGPyTorch models")

    def subset_output(self, idcs: List[int]) -> Model:
        raise NotImplementedError("Condition on observations is not yet supported for DagGPyTorch models")
