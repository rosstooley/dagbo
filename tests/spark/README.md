# Spark Demo
A demo of the code used to run the Spark experiment in my dissertation.

The demo does not require Spark because the execution of Spark is stubbed. Running the `bo_loop.py` demo does BO on fake (randomly generated) data.

## Overview
* `spark_dag.py` contains the DAG used in the Spark case study.
* `bo_loop.py` uses the Spark DAG in a Bayesian optimisation loop.

Run with: `python3 bo_loop.py`
