import argparse
import warnings
from typing import Union
from logging import INFO
from datasets import Dataset, DatasetDict
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr_datasets.partitioner import IidPartitioner


warnings.filterwarnings("ignore", category=UserWarning)

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
args = parser.parse_args()


# Define data partitioning related functions
def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


# Load (HIGGS) dataset and conduct partitioning
# We use a small subset (num_partitions=30) of the dataset for demonstration to speed up the data loading process.
partitioner = IidPartitioner(num_partitions=30)
fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
partition = fds.load_partition(node_id=args.partition_id, split="train")
partition.set_format("numpy")

# Train/test splitting
X_train,X_test,y_train,y_test = train_test_split(
    partition, test_fraction=0.2, seed=42
)



# Define and train XGBoost classifier
xgbr = xgb.XGBClassifier(verbosity=0)
xgbr.fit(X_train, y_train)

y_pred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("MSE ",mse)

print("RMSE ",mse**(1/2))

# Evaluate the model
train_score = xgbr.score(X_train, y_train)
test_score = xgbr.score(X_test, y_test)

print("Training score:", train_score)
print("Test score:", test_score)
