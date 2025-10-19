import os.path

import numpy as np
import pandas as pd
import argparse


def rmsle(predictions, actuals):
    rmsle_confirmed = np.sqrt(
        np.mean(
            (
                np.log1p(predictions["ConfirmedCases"])
                - np.log1p(actuals["ConfirmedCases"])
            )
            ** 2
        )
    )
    rmsle_fatalities = np.sqrt(
        np.mean(
            (np.log1p(predictions["Fatalities"]) - np.log1p(actuals["Fatalities"])) ** 2
        )
    )
    return (rmsle_confirmed + rmsle_fatalities) / 2


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="count")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

performance = rmsle(predictions, answers)

with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
