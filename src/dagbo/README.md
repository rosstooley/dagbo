# Repository structure
1. GPyTorch models:
    1. `ParametricMean` allows users to create parametric models in GPyTorch. They must sub-class ParametricMean to implement their model.
    2. `Node` is a semi-parametric model (Gaussian process which can use a ParametricMean)
    3. `Dag` allows users to create DAG models. They must sub-class Dag to implement their model.
2. BoTorch models:
    1. `DagGPyTorchModel` is a BoTorch model. Users must also sub DagGPyTorchModel to create BoTorch DAG models.
    2. `SampleAveragePosterior` is a BoTorch posterior. It wraps around a GPyTorch posterior and treats the outer batch of the MVN as Monte-Carlo samples. This allows users to create pseudo-deterministic posteriors of a Dag even though the Dag does internal sampling.
3. Inference and prediction helpers:
    1. `fit_dag.py` contains functions to run inference on Dag. This allows the model to be trained with SciPy, Torch or Pyro optimisers.
    2. `predict_dag.py` contains a small helper to predict from Dag, in case you don't need to wrap it in DagGPyTorchModel.
4. Ax utils
    1. `ax_utils.py` contains wrapper functions to get and fit a DAG model in Ax.

