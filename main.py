import pandas as pd
import numpy as np
import re
from contextlib import ExitStack
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans, cmeans_predict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
            proc.writelines(dataset)


def import_datasets() -> tuple:
    cols = pd.read_csv(datasets[0], nrows=1).columns
    return tuple(pd.read_csv(file, usecols=cols[:-1]) for file in datasets)


def read_diagnoses() -> tuple:
    cols = pd.read_csv(datasets[0], nrows=1).columns
    diagnoses = tuple()
    for dataset in datasets:
        read = pd.read_csv(dataset, usecols=cols[-1:None])
        diagnoses += (np.array([*map(lambda x: 1 if x == "Yes" else 0,
                                     read.values)]),)
    return diagnoses


def perform_crisp_clustering(training: np.array, test: np.array,
                             clusters: int, *args) -> tuple:
    kmeans = KMeans(clusters)
    kmeans.fit(training)
    return kmeans.labels_, kmeans.predict(test)


def perform_fuzzy_clustering(training: np.array, test: np.array,
                             clusters: int, m: int) -> tuple:
    center, train_labels = cmeans(training.T, clusters, m, 0.005, 1000)[0:2]
    test_labels = cmeans_predict(test.T, center, 2, 0.005, 1000)[0]
    return *(np.argmax(label, 0) for label in [train_labels, test_labels]),


def main():
    preprocess_datasets()
    training_set, test_set = import_datasets()
    algorithms = [perform_crisp_clustering, perform_fuzzy_clustering]
    diagnoses = read_diagnoses()
    for algorithm in algorithms:
        result = algorithm(training_set.values, test_set.values, 2, 2)
        print([sum(res) for res in [x == y for x, y in
                                    zip(result, diagnoses)]])

    # # pca.fit(test)
    # # test = pca.transform(test)
    # # pca.fit(train)
    # # train = pca.transform(train)
    # pca = PCA(2)
    # # pca.fit(uu)


if __name__ == "__main__":
    main()
