from torch import Tensor, cat, isnan
from botorch.models.model import Model
from ax.storage.runner_registry import register_runner
from .node import Node
from .dag import Dag
from .fit_dag import fit_dag, fit_node_with_mcmc, fit_node_with_scipy, fit_node_with_torch
from typing import Callable, List, Any, Optional, Dict

# These classes can be serialised by Ax
class FitNodeWithMcmc:
    def __call__(self, model: Node, *args: Any, **kwds: Any) -> Any:
        fit_node_with_mcmc(model, *args, **kwds)

class FitNodeWithTorch:
    def __call__(self, model: Node, *args: Any, **kwds: Any) -> Any:
        fit_node_with_torch(model, *args, **kwds)

class FitNodeWithSciPy:
    def __call__(self, model: Node, *args: Any, **kwds: Any) -> Any:
        fit_node_with_scipy(model, *args, **kwds)

class AxDagModelConstructor:
    def __init__(self, delayed_model_init: Callable[[List[str], List[str], Tensor, Tensor, int], Dag],
                 train_input_names: List[str], train_target_names: List[str], 
                 node_optimizer: Callable[[Node, Any], None] = FitNodeWithSciPy(),
                 num_samples: int = 128, **kwargs: Any):
        """
        Args:
            model_cls: A model extending both Dag and DagGPyTorchModel
            train_input_names / train_target_names: see Dag __init__ documentation
            node_optimizer: an optimizer to fit each node of the the DAG model
            kwargs: are forwarded to the node optimizer
        """
        if not (sorted(train_target_names) == train_target_names):
            raise RuntimeError("Ax requires that targets are in sorted order.")
        self.delayed_model_init = delayed_model_init
        self.train_input_names = train_input_names
        self.train_target_names = train_target_names
        self.node_optimizer = node_optimizer
        self.num_samples = num_samples
        self.kwargs = kwargs
    
    # FYI: Ax puts inputs in order defined by SearchSpace
    # FYI: Ax requires targets in alphabetical order
    def __call__(self, Xs: List[Tensor], Ys: List[Tensor], 
                 Yvars: List[Tensor], task_features: List[int],
                 fidelity_features: List[int], metric_names: List[str], 
                 state_dict: Optional[Dict[str, Tensor]] = None, **kwargs: Any) -> Model:
        """Instantiates and fits a DagGPyTorchModel using the given data.

        Args:
            Xs: List of X data, one tensor per outcome.
            Ys: List of Y data, one tensor per outcome.
            Yvars: List of observed variance of Ys.
            task_features: List of columns of X that are tasks.
            fidelity_features: List of columns of X that are fidelity parameters.
            metric_names: Names of each outcome Y in Ys.
            state_dict: If provided, will set model parameters to this state
                dictionary. Otherwise, will fit the model.
            refit_model: Flag for refitting model.

        Returns:
            A fitted GPyTorchModel.
        """
        if len(Yvars) > 0 and any([not all(isnan(t)) for t in Yvars]):
            raise RuntimeError("Yvars not supported by DagGPyTorch model.")
        if len(task_features) > 0:
            raise RuntimeError("Task features not supported by DagGPyTorch model.")
        if len(fidelity_features) > 0:
            raise RuntimeError("Fidelity features not supported by DagGPyTorch model.")
        train_inputs = Xs[0]
        named_Ys = {k:v for k,v in zip(metric_names, Ys)}
        ordered_Ys = [named_Ys[name] for name in self.train_target_names]
        train_targets = cat(ordered_Ys, dim=-1)
        model = self.delayed_model_init(self.train_input_names, self.train_target_names, train_inputs, train_targets, self.num_samples)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            fit_dag(model, self.node_optimizer, **kwargs)
        return model

def register_runners():
    register_runner(FitNodeWithMcmc)
    register_runner(FitNodeWithTorch)
    register_runner(FitNodeWithSciPy)
    register_runner(AxDagModelConstructor)

