# -*- UTF-8 -*-
import numpy as np
import pandas as pd

def submit(submit_file):
    submit_df = pd.read_csv(submit_file, encoding="utf-8")
    answer_df = pd.read_csv("data/answer.csv", encoding="utf-8")
    counter = 0
    for i in range(5000):
        if submit_df.iloc[i, 1] == answer_df.iloc[i, 1]:
            counter += 1

    print(counter / 5000.0)

def submit2(submit_file):
    submit_df = pd.read_csv(submit_file, encoding="utf-8")
    answer_df = pd.read_csv("data/answer.csv", encoding="utf-8")


if __name__ == '__main__':
    submit("submission.csv")