from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from torch import Size
from .parametric_mean import ParametricMean
from .node import Node
from .tensor_dict_conversions import pack_to_tensor, unpack_to_dict
from typing import Iterator, List, Optional, Union
from warnings import warn

class Dag(Module):
    """
    The DAG is a GPyTorch model with a directed, acyclic graph of sub-models. To
    create a DAG: subclass this class and implement define_dag using
    register_input and register_metric to "wire up" the DAG.

    DAG "inputs" are configurable parameters of the experiment. For example, when
    benchmarking a Java program we might configure the heap size and the number
    of CPUs.
    
    DAG "metrics" are the measurable outcomes of the experiment. For example,
    when benchmarking a Java program we might measure the total program time
    and also garbage-collection time.

    In the DAG model, all metrics are given their own sub-model which learns to
    predict the value of the metric. We can choose the inputs to each sub-model
    to be real inputs or to be other metrics.

    todo: discuss form of sub-models
    
    The sub-models are trained independently. Using the running example, the GC
    time of the Java program will be trained separately to the total time of the
    program. I.e.: 
    the GC time will be trained on data {x: (heap_size, num_cpus), y: gc_time}
    and total time will be trained on data {x: (num_cpus, gc_time), y: total_time}

    During prediction, all sub-models are dependent on their inputs. I.e.:
    GC time uses x=(heap_size, num_cpus) to predict gc_time
    then total time uses y=(num_cpus, gc_time) to predict total_time
    """

    def __init__(self, train_input_names: List[str], train_target_names: List[str],
                 train_inputs: Tensor, train_targets: Tensor):
        """
        Args:
            train_input_names: a d-length List of the names of each input
                defining the order of the inputs in the innermost
                dimension of train_inputs.
            train_target_names: a m-length List of the names of each target
                defining the order of the inputs in the innermost
                dimension of train_targets.
            train_inputs: A batch_shape*q*d-dim Tensor of the training inputs
                The innermost dim, dim=-1, must follow the same order as
                train_input_names as this is how the data is split up around
                the DAG.
            train_targets: A batch_shape*q*m-dim Tensor of the training targets
                The innermost dim, dim=-1, must follow the same order as
                train_targets_names as this is how the data is split up around
                the DAG.
        """
        super().__init__()
        self._num_outputs = len(train_target_names)
        batch_shape = train_inputs.shape[:-2]
        self.input_names = train_input_names
        self.target_names = train_target_names
        # data is explicitly un-batched in DAG since it will be split up for each sub-model
        # it will then be re-batched before adding it to the submodel
        self.train_inputs = unpack_to_dict(self.input_names, train_inputs)
        self.train_targets = unpack_to_dict(self.target_names, train_targets)
        self.registered_input_names = []
        # nodes themselves will be accessed using self.named_children
        self.registered_target_names = []
        self.define_dag(batch_shape)
        self._error_unspecified_outputs()
        self._warn_unused_inputs()
        self.to(train_inputs)

    def define_dag(self, batch_shape: Size) -> None:
        """
        Must be implemented in subclass
        Creates the nodes and edges of the DAG
        # todo: example
        """
        raise NotImplementedError()

    def register_input(self, name: str) -> str:
        self._error_missing_names([name])
        self.registered_input_names.append(name)
        return name

    def register_metric(self, name: str, inputs: List[str],
                        mean: Optional[Union[Mean, ParametricMean]] = None,
                        covar: Optional[Kernel] = None,
                        likelihood: Optional[_GaussianLikelihoodBase] = None) -> str:
        # todo: using "inspect" to add error for ParametricMeans with fwd arguments not in "inputs"
        self._error_missing_names([name] + inputs)
        self._error_unregistered_inputs(inputs, name)
        X_from_inputs = {k:v for k,v in self.train_inputs.items() if k in inputs}
        X_from_outputs = {k:v for k,v in self.train_targets.items() if k in inputs}
        X = pack_to_tensor(inputs, {**X_from_inputs, **X_from_outputs})
        y = self.train_targets[name]
        node = Node(inputs, name, X, y, mean)
        self.add_module(name, node)
        self.registered_target_names.append(node.output_name)
        return name

    def _nodes_order(self, order) -> Iterator[Node]:
        """Returns: iterator over DAG's nodes in `order` order"""
        for name in order:
            yield getattr(self, name)

    def nodes_dag_order(self) -> Iterator[Node]:
        """Returns: iterator over DAG's nodes in the order specified in define_dag"""
        return self._nodes_order(self.registered_target_names)

    def nodes_output_order(self) -> Iterator[Node]:
        """Returns: iterator over DAG's nodes in the order specified in train_target_names"""
        return self._nodes_order(self.target_names)

    def forward(self, test_inputs: Tensor) -> Union[MultivariateNormal, MultitaskMultivariateNormal]:
        """
        This is only used for prediction, since the individual nodes
        are trained independently
        Args:
            test_inputs: batch_shape*q*d-dim tensor
        """
        # since the nodes must be registered in topological order
        #   then we can do the predictions in the same order and use 
        #   test_inputs_d to store them
        # also need to pack into tensors before passing to sub-models
        
        test_inputs_d = unpack_to_dict(self.input_names, test_inputs)
        test_metrics_d = {}
        for node in self.nodes_dag_order():
            node_inputs_d = {k:v for k,v in test_inputs_d.items() if k in node.input_names}
            node_inputs = pack_to_tensor(node.input_names, node_inputs_d)
            # mvn: batch_shape MVN with q points considered jointly
            mvn = node(node_inputs)
            test_metrics_d[node.output_name] = mvn
            prediction = mvn.rsample()
            test_inputs_d[node.output_name] = prediction
        if len(self.target_names) > 1:
            # mvns must be in the expected output order
            mvns = [test_metrics_d[metric] for metric in self.target_names]
            return MultitaskMultivariateNormal.from_independent_mvns(mvns)
        else:
            return test_metrics_d[self.target_names[0]]

    def _error_missing_names(self, names):
        missing_names = set(names).difference(self.input_names).difference(self.target_names)
        if missing_names:
            raise NameError(str(missing_names) + " defined in DAG but not declared in train_input_names or train_target_names.")

    def _error_unregistered_inputs(self, input_names, output_name):
        unregisted_inputs = set(input_names).difference(self.registered_input_names).difference(self.registered_target_names)
        if unregisted_inputs:
            raise NameError(str(unregisted_inputs) + " defined as input to " + output_name + " before being registered.")

    def _error_unspecified_outputs(self):
        unspecified_outputs = set(self.target_names).difference(self.registered_target_names)
        if unspecified_outputs:
            raise RuntimeError("All train_targets_names must be specified in DAG model, but "
                               + str(unspecified_outputs) + " are not.")

    def _warn_unused_inputs(self):
        missing_fields = set(self.input_names).difference(self.registered_input_names)
        if missing_fields:
            warn(str(missing_fields) + " defined in train_input_names but not used in DAG.")

