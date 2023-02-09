import pandas as pd
from datasets import load_dataset
import os
import re
from sklearn.model_selection import train_test_split


def load_train_data(hparams):
    dfs = []
    names = os.listdir(hparams.data_dir)
    for name in names:
        path_to_data = os.path.join(hparams.data_dir, name)
        temp_df = pd.read_csv(path_to_data)
        dfs.append(temp_df)
    df = pd.concat(dfs)
    df.columns = ["labels", "text"]

    mapping = {
        "how to do a task": 0,
        "unable to complete a task": 1,
        "something isnt working how it should": 2,
    }

    df["labels"] = df["labels"].replace(mapping)
    df["text"] = df["text"].apply(lambda x: re.sub(r"=+", " ", x))

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["labels"])
    val_df, test_df = train_test_split(val_df, test_size=0.1, stratify=val_df["labels"])

    train_df = train_df.iloc[0:1000]
    val_df = val_df.iloc[0:100]
    return train_df, val_df
