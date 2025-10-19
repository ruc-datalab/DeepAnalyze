import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="target")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

answers.sort_values(by="ID")
predictions.sort_values(by="ID")


def rmsle(actual, predicted):
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    actual (list or np.array): Array of actual target values.
    predicted (list or np.array): Array of predicted target values.

    Returns:
    float: The RMSLE score.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate the log of actual and predicted values
    log_actual = np.log(actual + 1)
    log_predicted = np.log(predicted + 1)

    # Calculate the squared differences
    squared_diff = (log_actual - log_predicted) ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Return the square root of the mean squared differences
    rmsle_value = np.sqrt(mean_squared_diff)
    return rmsle_value


performance = rmsle(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
