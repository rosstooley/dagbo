from ax.core.simple_experiment import SimpleExperiment
from ax.core.search_space import SearchSpace
from ax.core.parameter import RangeParameter, ChoiceParameter, ParameterType
from ax.core.parameter_constraint import ParameterConstraint
from ax.modelbridge.factory import Models, get_botorch
from ax.storage.runner_registry import register_runner
from ax.utils.common.constants import Keys
from ax import save

from dagbo.ax_utils import AxDagModelConstructor, register_runners

from spark_dag import SparkDag

"""
An integration test of the Spark DAG in a tuning loop
1. Register Spark model with Ax
2. Define search space and metrics
3. Create experiment controller
4. Initialise with bootstrap evaluations
5. Run the tuner
  a. Define and fit the Spark DAG model
  b. Optimise the acquisition
  c. Evaluate next configuration (call to Spark is stubbed)
  d. Save the result and repeat
"""

# MODEL REGISTRY (for saving results)
register_runners()
class DelayedSparkDagInit:
    def __call__(self, train_input_names, train_target_names, train_inputs, train_targets, num_samples):
        return SparkDag(train_input_names, train_target_names, train_inputs, train_targets, num_samples)
register_runner(DelayedSparkDagInit)

# SEARCH SPACE
executor_cores = RangeParameter("spark.executor.cores", ParameterType.INT, lower=1, upper=8)
executor_memory = RangeParameter("spark.executor.memory", ParameterType.INT, lower=512, upper=14336)
task_cpus = RangeParameter("spark.task.cpus", ParameterType.INT, lower=1, upper=8)
memory_fraction = RangeParameter("spark.memory.fraction", ParameterType.FLOAT, lower=0.01, upper=0.99)
shuffle_compress = ChoiceParameter("spark.shuffle.compress", ParameterType.BOOL, values=[False, True], is_ordered=True)
shuffle_spill_compress = ChoiceParameter("spark.shuffle.spill.compress", ParameterType.BOOL, values=[False, True], is_ordered=True)
# order of parameters (configurable variables) is arbitrary be must be consistent throughout program
parameters = [executor_cores, executor_memory, task_cpus, memory_fraction, shuffle_compress, shuffle_spill_compress]

task_cpus_lt_executor_cores = ParameterConstraint(constraint_dict={"spark.task.cpus": 1., "spark.executor.cores": -1.}, bound=0.)

parameter_constraints = [task_cpus_lt_executor_cores]

search_space = SearchSpace(parameters, parameter_constraints)

# METRICS
# order of metric names must be in alphabetical order
metric_keys = sorted([
    "num_executors",
    "num_tasks_per_executor",
    "concurrent_tasks",
    "disk_bytes_spilled_0",
    "executor_cpu_time_0",
    "jvm_gc_time_0",
    "executor_noncpu_time_0",
    "duration_0",
    "disk_bytes_spilled_2",
    "executor_cpu_time_2",
    "jvm_gc_time_2",
    "executor_noncpu_time_2",
    "duration_2",
    "throughput_from_first_job",
])

# EVALUATION FUNCTION
def get_eval_fun():
    """
    Returns the function which evaluates Spark on a new configuration.
    This is a stub for the unit tests (which don't have Spark integration).
    """
    from random import random
    return lambda _: {k:(random(), float('nan')) for k in metric_keys}

# EXPERIMENT CONTROLLER
simple_exp = SimpleExperiment(
    search_space=search_space,
    objective_name="throughput_from_first_job",
    name="spark_exp_example",
    evaluation_function=get_eval_fun()
)

num_bootstrap = 2
num_trials = 4

# BOOTSTRAP EVALUATIONS
sobol = Models.SOBOL(simple_exp.search_space)
simple_exp.new_batch_trial(generator_run=sobol.gen(num_bootstrap))
data=simple_exp.eval()

# TUNING EVALUATIONS
arc_samples = 4
model_constructor = AxDagModelConstructor(
    DelayedSparkDagInit(),
    list(simple_exp.search_space.parameters.keys()),
    metric_keys,
    num_samples=arc_samples
)
for _ in range(num_trials):
    # model fitting
    model = get_botorch(
        search_space=simple_exp.search_space,
        experiment=simple_exp,
        data=data,
        model_constructor=model_constructor
    )

    # reducing peak memory
    num_points = 1000
    batch_size = num_points // arc_samples
    model_gen_options = {Keys.OPTIMIZER_KWARGS: {"batch_limit": batch_size}}

    # acquisition optimisation
    simple_exp.new_trial(model.gen(1, model_gen_options=model_gen_options))

    # evaluation of next configuration
    data=simple_exp.eval()

    # saving results
    save(simple_exp, "spark_example_exp.json")
