import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="cost")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = np.sqrt(
    mean_squared_log_error(answers[args.value], predictions[args.value])
)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
