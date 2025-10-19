import os.path
import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="claim")

args = parser.parse_args()


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def mean_column_wise_rmsle(true_df, pred_df):
    assert true_df.shape == pred_df.shape
    num_columns = true_df.shape[1]
    rmsle_values = []

    for column in true_df.columns:
        if column != "date_time":  # Skip the date_time column
            rmsle_values.append(rmsle(true_df[column].values, pred_df[column].values))

    return np.mean(rmsle_values)


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)
answers.sort_values(by=["date_time"])
predictions.sort_values(by=["date_time"])


performance = mean_column_wise_rmsle(answers, predictions)


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
