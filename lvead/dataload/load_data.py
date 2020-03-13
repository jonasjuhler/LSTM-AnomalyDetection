
import os

import numpy as np


def load_data(dataset, basepath='../../'):
    if dataset not in ['KDD99', 'Arrhythmia']:
        filepath_train = os.path.join(
            basepath, 'data', dataset, f'{dataset}_TRAIN.txt'
            )
        filepath_test = os.path.join(
            basepath, 'data', dataset, f'{dataset}_TEST.txt'
            )
        train_data = np.loadtxt(filepath_train)
        test_data = np.loadtxt(filepath_test)
    elif dataset == 'Arrythmia':
        pass
    else:
        raise FileNotFoundError(
            'KDD99 dataset could not be found in directory'
            )
    return train_data, test_data
