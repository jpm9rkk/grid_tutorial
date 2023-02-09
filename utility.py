import pandas as pd
from datasets import load_dataset


def load_train_data():
    train_df = pd.DataFrame(load_dataset("ag_news", split="train[:20%]")).rename(
        columns={"label": "labels", "text": "full_text"}
    )
    val_df = pd.DataFrame(load_dataset("ag_news", split="test[:20%]")).rename(
        columns={"label": "labels", "text": "full_text"}
    )
    return train_df, val_df
