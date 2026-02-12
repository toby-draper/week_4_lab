# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# Load csv into df:
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# %%
# drop unncecessary columns from df:
def keep_columns_by_index(df: pd.DataFrame, to_keep: list[int]) -> pd.DataFrame:
    return df[df.columns[to_keep]]

# %%
# one hot encode a column and transform it into binary:
def one_hot_binary(df: pd.DataFrame, col: str, positive_level: str) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=[col])
    target_col = f"{col}_{positive_level}"
    df[target_col] = df[target_col].astype(int)

    for c in df.columns:
        if c.startswith(f"{col}_") and c != target_col:
            df.drop(columns=c, inplace=True)

    return df


# %%
# compmute prevalence of a binary variable:
def compute_prevalence(df: pd.DataFrame, binary_col: str) -> float:
    prevalence = df[binary_col].mean() * 100
    print(prevalence)
    return prevalence

# %%
# normalize continuous variables from 0 to 1:
def minmax_scale_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = MinMaxScaler().fit_transform(df[[col]])
    return df

# %%
# split df into train and test sets:
def split_train_test(df: pd.DataFrame, target_col: str, train_size: float = 0.7):
    train, test = train_test_split(df, train_size=train_size, stratify=df[target_col])
    print(train.shape)
    print(test.shape)
    return train, test

# %%
# split test set into tune and test sets:
def split_tune_test(test_df: pd.DataFrame, target_col: str, train_size: float = 0.5):
    tune, test = train_test_split(test_df, train_size=train_size, stratify=test_df[target_col])
    print(tune.shape)
    print(test.shape)
    return tune, test

# %%
