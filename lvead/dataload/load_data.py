import os

import numpy as np


def load_data(dataset, basepath="../../"):
    if dataset not in ["KDD99", "Arrhythmia"]:
        filepath_train = os.path.join(
            basepath, "data", dataset, f"{dataset}_TRAIN.txt")
        filepath_test = os.path.join(
            basepath, "data", dataset, f"{dataset}_TEST.txt")
        train_data = np.loadtxt(filepath_train)
        test_data = np.loadtxt(filepath_test)
    elif dataset == "Arrhythmia":
        filepath_train = os.path.join(
            basepath, "data", dataset, "arrhythmia.data")
        train_data = np.genfromtxt(filepath_train, delimiter=',')
        test_data = np.empty(0)
    else:
        raise FileNotFoundError(
            f"{dataset} dataset could not be found in directory,\
            \nData has not yet been downloaded due to size constraints")
    return train_data, test_data


if __name__ == '__main__':
    datasets = [
        "Wafer",
        "TwoLeadECG",
        "ToeSegmentation2",
        "KDD99",
        "MoteStrain",
        "Herring",
        "ItalyPowerDemand",
        "ECGFiveDays",
        "GunPointAgeSpan",
        "Arrhythmia"
    ]

    for dataset in datasets:
        try:
            train, test = load_data(dataset)
            print(f'{dataset}:\nTrain data shape: ', train.shape)
            print('Test data shape: ', test.shape, '\n')
        except FileNotFoundError as err:
            print(f"FileNotFoundError: {err}\n")
