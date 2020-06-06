import pandas as pd
import numpy as np
import re
from contextlib import ExitStack
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans


raw_datasets = ["models/" + name for name in ["pima.tr", "pima.te"]]
datasets = ["models/" + name for name in ["training.csv", "test.csv"]]


def preprocess_datasets() -> None:
    with ExitStack() as stack:
        raws = [stack.enter_context(open(file, 'r')) for file in raw_datasets]
        processed = [stack.enter_context(open(file, 'w')) for file in datasets]
        for raw, proc in zip(raws, processed):
            dataset = raw.readlines()
            dataset = [re.sub(r"^ +", "", row) for row in dataset]
            dataset = [re.sub(r" +", ",", row) for row in dataset]
            dataset = [re.sub(r",[^,]+$", "\n", row) for row in dataset]
            proc.writelines(dataset)


def import_datasets() -> tuple:
    return tuple(pd.read_csv(dataset) for dataset in datasets)


def main():
    preprocess_datasets()
    training_set, test_set = import_datasets()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(training_set.values)
    kmeans.predict(test_set.values)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)


if __name__ == "__main__":
    main()
