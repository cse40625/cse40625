import pandas as pd

train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes'


def make_dataset():
    """Make the binary digits dataset."""
    df_train = pd.read_csv(train, header=None)
    df_test = pd.read_csv(test, header=None)
    df = pd.concat([df_train, df_test], axis=0)
    df = df.rename(columns={64: 'Class'})
    df = df.loc[(df.loc[:, 'Class'] == 1) | (df.loc[:, 'Class'] == 5)]
    df.loc[df.loc[:, 'Class'] == 1, 'Class'] = -1
    df.loc[df.loc[:, 'Class'] == 5, 'Class'] = 1
    df.to_csv('digits_binary.csv', index=False)

if __name__ == "__main__":
    make_dataset()
