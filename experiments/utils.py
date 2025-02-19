import torch
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from torch.autograd import Variable

def split_train_test(
        dataset: pd.core.frame.DataFrame,
        train_percentage: float
    ) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    """Split the dataset in trainset and testset.

    Args:
        dataset(pandas.core.frame.DataFrame): any dataset.
        train_percentage(float): split trainset and test according this parameter.

    Returns:
        a tuple(pandas.core.frame.DataFrame, pandas.core.frame.DataFrame) with the trainset and testset.
    """
    train_size = round(len(dataset) * train_percentage)
    train = dataset.iloc[:train_size]
    test = dataset.iloc[train_size:]
    return train, test

def normalize(normalizer, trainset, testset, features):
    train_scaled = pd.DataFrame(normalizer.fit_transform(trainset), columns=features)
    test_scaled = pd.DataFrame(normalizer.transform(testset), columns=features)
    return train_scaled, test_scaled

def create_sequence(
        dataset: pd.core.frame.DataFrame, 
        lookback: int,
        lookforward: int
    ) -> Tuple[np.array,np.array]:
    """Transform a time series into a prediction dataset.
    
    Args:
        dataset(pandas.core.frame.DataFrame): the time series where the first dimension is the timestep. Consider to normalize the dataset before to create the sequence.
        lookback(int): size of window for fixed input.

    Returns:
        a tuple of two np.arrays with X and y.
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset.iloc[i:i+lookback].to_numpy()
        target = dataset.iloc[i+lookback].to_numpy()
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

def create_tensor(data):
    return Variable(torch.Tensor(data))

def store_dataset_overview(title: str, ylabel: str, data: pd.core.frame.DataFrame, labels: list, filename: str):
    plt.figure(figsize=(16, 8))
    # plt.title(title, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.xticks(fontsize=28) 
    plt.plot(data, label=labels)
    plt.legend(fontsize=20)
    plt.savefig(filename)