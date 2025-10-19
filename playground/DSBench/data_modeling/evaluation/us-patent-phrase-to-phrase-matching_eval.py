import os.path

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import median_absolute_error
from scipy.stats import pearsonr


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="score")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

if predictions[args.value].isnull().any():
    print("Error: There are missing values in the score columns.")

performance, p_value = pearsonr(predictions[args.value], answers[args.value])

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
