from torch import Size, Tensor
from torch.nn.parameter import Parameter
from gpytorch import Module
from gpytorch.priors import Prior
from gpytorch.constraints.constraints import Interval
from .tensor_dict_conversions import unpack_to_dict
from typing import Dict, List

class ParametricMean(Module):
    def __init__(self, batch_shape: Size = Size([])):
        super().__init__()
        self.init_params(batch_shape)

    def register_bayesian_parameter(self, name: str, prior: Prior, constraint: Interval = None) -> None:
        """
        Helper class to quickly define parameter, prior and constraint
        Sets initial point value of the parameter to the mean of the prior
        """
        self.register_parameter(name, Parameter(prior.mean))
        self.register_prior(name + "_prior", prior, name)
        if constraint is not None:
            self.register_constraint(name, constraint)

    def init_params(self, batch_shape: Size = Size([])) -> None:
        """
        Must be implemented in subclass
        Creates the parameters, and their constraints and priors

        e.g.
        self.register_bayesian_parameter("m_x1", NormalPrior(zeros(*batch_shape, 1), 1.))
        self.register_bayesian_parameter("m_x2", NormalPrior(zeros(*batch_shape, 1), 1.))
        self.register_bayesian_parameter("c", NormalPrior(zeros(*batch_shape, 1), 1.))
        """
        raise NotImplementedError()

    def forward(self, **kwargs: Dict[str, Tensor]) -> Tensor:
        """
        Must be implemented in subclass

        Returns the posterior of the parametric mean function at `kwargs`
        conditioned on the training data

        Use named args to access named inputs; e.g. in
            forward(self, distance, time, **kwargs)
            distance and time will be Tensors containing
            the input data named `distance` and `time` respectively
        All node inputs specified in DAG are available as kwargs.

        This does come at a performance penalty because batches cannot be exploited.
        If each input has the same relationship with the output (e.g. "linear") then
        use a batch kernel subclassing "Mean" such as "LinearMean"

        Args:
            kwargs: each named arg is a batch_shape * q-dim Tensor
                containing the input data for one input site.

        e.g.
        return self.m_x1 * x1 + self.m_x2 * x2 + c
        """
        raise NotImplementedError()

    def __call__(self, x: Tensor, field_names: List[str]) -> Tensor:
        """
        Converts from b*d-dim Tensor of unnamed inputs to Dict[str, b-dim Tensor] of size d
        Args:
            x: b*d-dim Tensor of unnamed inputs.
            field_names: d-length of input names in the same order as the inner dim of x
        """
        x_d = unpack_to_dict(field_names, x)
        return super().__call__(**x_d)

