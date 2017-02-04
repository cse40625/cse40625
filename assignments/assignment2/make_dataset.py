import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes'


def make_dataset():
    """Make the binary digits dataset."""
    df_train = pd.read_csv(train, header=None)
    df_test = pd.read_csv(test, header=None)
    df = pd.concat([df_train, df_test], axis=0)
    df = df.rename(columns={64: 'Class'})
    df = df.ix[(df.ix[:, -1] == 1) | (df.ix[:, -1] == 5)]
    df.ix[df.ix[:, -1] == 1, -1] = -1
    df.ix[df.ix[:, -1] == 5, -1] = 1
    df.to_csv('digits_binary.csv', index=False)

if __name__ == "__main__":
    make_dataset()
