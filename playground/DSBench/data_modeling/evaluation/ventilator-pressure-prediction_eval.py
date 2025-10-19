import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="pressure")

args = parser.parse_args()


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

answers.sort_values(by=["id"])
predictions.sort_values(by=["id"])
performance = mean_absolute_error(answers[args.value], predictions[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
