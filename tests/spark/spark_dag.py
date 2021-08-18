from typing import List
from torch import Size, Tensor
from dagbo.dag import Dag
from dagbo.dag_gpytorch_model import DagGPyTorchModel

class SparkDag(Dag, DagGPyTorchModel):
    """DAG model for the SQL/Aggregation HiBench Spark benchmark"""

    def __init__(self, train_input_names: List[str], train_target_names: List[str], train_inputs: Tensor, train_targets: Tensor, num_samples: int):
        super().__init__(train_input_names, train_target_names, train_inputs, train_targets)
        # required for all classes that extend SparkDag
        self.num_samples = num_samples

    def define_dag(self, batch_shape: Size = Size([])) -> None:
        sec = self.register_input("spark.executor.cores")
        stc = self.register_input("spark.task.cpus")
        sem = self.register_input("spark.executor.memory")
        smf = self.register_input("spark.memory.fraction")
        ssc = self.register_input("spark.shuffle.compress")
        sssc = self.register_input("spark.shuffle.spill.compress")

        ne = self.register_metric("num_executors", [sec, sem])
        tpe = self.register_metric("num_tasks_per_executor", [sec, stc])
        ct = self.register_metric("concurrent_tasks", [ne, tpe])
       
        # SQL/Aggregation benchmark has two stages, so there are two of these metrics
        # first stage
        dbs0 = self.register_metric("disk_bytes_spilled_0", [sec, sem, smf, ssc])
        cpu0 = self.register_metric("executor_cpu_time_0", [dbs0, ct, ssc])
        jgc0 = self.register_metric("jvm_gc_time_0", [sem, tpe])
        ncpu0 = self.register_metric("executor_noncpu_time_0", [ct, jgc0])
        d0 = self.register_metric("duration_0", [cpu0, ncpu0])

        # second stage
        dbs2 = self.register_metric("disk_bytes_spilled_2", [sec, sem, smf, ssc])
        cpu2 = self.register_metric("executor_cpu_time_2", [dbs2, ct])
        jgc2 = self.register_metric("jvm_gc_time_2", [sem, tpe])
        ncpu2 = self.register_metric("executor_noncpu_time_2", [ct, jgc2])
        d2 = self.register_metric("duration_2", [cpu2, ncpu2])

        self.register_metric("throughput_from_first_job", [d0, d2, sssc])
