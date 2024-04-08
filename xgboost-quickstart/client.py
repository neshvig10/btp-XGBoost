import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from typing import Union
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import flwr as fl
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
def custom_train_test_split(data: pd.DataFrame, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_data, valid_data = sklearn_train_test_split(data, test_size=test_fraction, random_state=seed)  # Use renamed function
    num_train = len(train_data)
    num_val = len(valid_data)
    return train_data, valid_data, num_train, num_val


def transform_dataset_to_dmatrix(data: pd.DataFrame, label_column: str) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    X = data.drop(columns=label_column).values
    y = data[label_column].values
    return xgb.DMatrix(X, label=y)


# Load CSV dataset
data = pd.read_csv("./btp.csv")

# Check if the required columns exist
required_columns = ['CMD', 'type', 'MEM']
missing_columns = set(required_columns) - set(data.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

le = LabelEncoder()
data["CMD"] = le.fit_transform(data["CMD"])
data["type"] = le.fit_transform(data["type"])
data = data.dropna()

# Train/test splitting
train_data, valid_data, num_train, num_val = custom_train_test_split(
    data, test_fraction=0.2, seed=42
)

# Reformat data to DMatrix for xgboost
train_dmatrix = transform_dataset_to_dmatrix(train_data, 'MEM')
valid_dmatrix = transform_dataset_to_dmatrix(valid_data, 'MEM')

# Hyper-parameters for xgboost training
num_local_round = 1
params = {
    "objective": "reg:squarederror",  # Use squared error for regression
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "rmse",  # Use RMSE for evaluation
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(self):
        self.bst = None

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
            )
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)

        # Update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        local_model = self.bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        rmse = float(eval_results.split("\t")[1].split(":")[1])

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=num_val,
            metrics={"RMSE": rmse},
        )


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=XgbClient().to_client())