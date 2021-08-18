from torch import Tensor, no_grad
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.settings import fast_pred_var
from .dag import Dag

def predict(dag_model: Dag, test_data: Tensor) -> MultitaskMultivariateNormal:
    """
    Can use this little helper function to predict from a Dag without
    wrapping it in a DagGPyTorchModel.
    """
    dag_model.eval()
    with no_grad(), fast_pred_var():
        return dag_model(test_data)

