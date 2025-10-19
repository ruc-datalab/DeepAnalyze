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


def gini(actual, pred):
    assert len(actual) == len(pred)
    all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))])
    all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)


def normalized_gini(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = normalized_gini(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
