# Loading the train, test data in csv form locally.
# Checking the shape and column values to make sure the files have correctly been pulled.

import pandas as pd
import os


def join_path(folder, file):
    dir_path = os.getcwd()
    path = dir_path + '/titanic/' + folder + '/' + file # for local
    #path = dir_path + '/' + 'titanic/' + folder + '/' + file
    return path


def read_train():
    train = pd.read_csv(join_path("data", "train.csv"), index_col="PassengerId")
    print("train shape: ", train.shape)
    print("train columns: ", train.columns.values)
    return train


def read_test():
    test = pd.read_csv(join_path("data", "test.csv"), index_col="PassengerId")
    print("test shape: ", test.shape)
    print("test columns: ", test.columns.values)
    return test