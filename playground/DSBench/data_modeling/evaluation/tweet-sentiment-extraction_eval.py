import os.path

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="yield")

args = parser.parse_args()


# 计算 Jaccard 相似度
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)

# 提取数据
y_true = answers["selected_text"].values
y_pred = predictions["selected_text"].values

n = len(answers)
performance = sum(jaccard(y_true[i], y_pred[i]) for i in range(n)) / n


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
