from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from pyro.infer import NUTS, MCMC
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from .node import Node
from .dag import Dag
from typing import Any, Callable

def get_pyro_model(model: Node, mll: ExactMarginalLogLikelihood):
    def pyro_model(X, y):
        model.pyro_sample_from_prior()
        output = model(X)
        mll.pyro_factor(output, y) # loss
        return y
    return pyro_model

def fit_node_with_mcmc(model: Node, num_samples: int, warmup_steps: int, **kwargs: Any) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    nuts = NUTS(get_pyro_model(model, mll))
    mcmc = MCMC(nuts, num_samples, warmup_steps)
    mcmc.run(*model.train_inputs, model.train_targets)
    # store the samples in the node by calling "pyro_load_from_samples"
    # this follows the same pattern as the torch / scipy optimisers which set the parameter values equal to their optimised value
    # the only difference is that they are set to a batch of values, one for each MCMC value
    model.pyro_load_from_samples(mcmc.get_samples())

def fit_node_with_torch(model: Node, **kwargs: Any) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll, fit_gpytorch_torch)

def fit_node_with_scipy(model: Node, **kwargs: Any) -> None:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    fit_gpytorch_model(mll)

def fit_dag(dag_model: Dag, node_optimizer: Callable[[Node, Any], None] = fit_node_with_scipy, **kwargs: Any):
    for node in dag_model.nodes_output_order():
        node_optimizer(node, **kwargs)

