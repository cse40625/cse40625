import numpy as np
import pandas as pd
from sklearn.datasets import load_digits


def make_dataset():
    """Make the binary digits dataset."""
    digits = load_digits(n_class=2)
    df = pd.DataFrame(np.column_stack((digits.data, digits.target)))
    df = df.rename(columns={64: 'Class'})
    df.loc[df.loc[:, 'Class'] == 0, 'Class'] = -1
    df.to_csv('digits_binary.csv', index=False)


if __name__ == "__main__":
    make_dataset()
