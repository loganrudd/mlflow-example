"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id
import argparse

def workflow(split_prop):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        load_raw_data_run = mlflow.run(".", "load_raw_data")
        load_raw_data_run_id = mlflow.tracking.MlflowClient()\
            .get_run(load_raw_data_run.run_id)
        loans_csv_uri = os.path.join(load_raw_data_run_id.info.artifact_uri,
                                     "loans-raw-csv-dir")
        '''
        etl_data_run = mlflow.run(".", "etl_data",
                                  parameters={"loans_csv_uri": loans_csv_uri})
        etl_data_run_id = mlflow.tracking.MlflowClient() \
            .get_run(etl_data_run.run_id)
        loans_parquet_uri = os.path.join(etl_data_run_id.info.artifact_uri,
                                         "loans-processed-parquet-dir")
        '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_prop', default=0.8, type=float)
    args = parser.parse_args()

    workflow(args.split_prop)
