import os.path

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="Transported")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = (predictions[args.value] == answers[args.value]).mean()

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
