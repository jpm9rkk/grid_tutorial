#
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
import wandb
from datasets import load_dataset

from loader import Encoder
from loader import MultiClassDataset
from loader import MultiClassDataModule
from model import TransformerForSequenceClassification


os.chdir("/Users/james.morrissey/Grid/cx_data_science/solvvy_intent/")

ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# df = pd.read_csv("./data/zero_shot_labels_0_1000.csv")
# df_2 = pd.read_csv("./data/zero_shot_labels_1000_2000.csv")
# df_3 = pd.read_csv("./data/zero_shot_labels_2000_3000.csv")
# df = pd.concat([df, df_2, df_3])

# df["seqs"] = df["seqs"].apply(lambda x: x.lower())
# df["seqs"] = df["seqs"].apply(lambda x: re.sub(r"=+", " ", x))
# df["preds"] = df["preds"].apply(lambda x: x.replace(" ", "_"))
# df["preds"] = df["preds"].replace(
#     {
#         "how_to_do_a_task": 0,
#         "unable_to_complete_a_task": 1,
#         "something_isnt_working_how_it_should": 2,
#     }
# )

# df.columns = ["labels", "full_text"]
# train_df, val_df = train_test_split(
#     df, test_size=0.1, random_state=42, stratify=df.labels
# )
train_df = pd.DataFrame(load_dataset("ag_news", split="train[:20%]")).rename(
    columns={"label": "labels", "text": "full_text"}
)
val_df = pd.DataFrame(load_dataset("ag_news", split="test[:20%]")).rename(
    columns={"label": "labels", "text": "full_text"}
)

breakpoint()

encoder = Encoder(tokenizer)
dm = MultiClassDataModule(train_df, val_df, tokenizer, encoder)
dm.setup()
sample_batch = next(iter(dm.train_dataloader()))

NUM_EPOCHS = 1
TRAIN_BATCHES = len(dm.train_dataloader())
TRAIN_STEPS = NUM_EPOCHS * TRAIN_BATCHES
model_hparams = {"num_steps": TRAIN_STEPS, "num_labels": 4}
model = TransformerForSequenceClassification(**model_hparams)

wandb_logger = WandbLogger(
    project="grid-classification",
)
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    # logger=wandb_logger,
    callbacks=[
        StochasticWeightAveraging(swa_lrs=1e-2),
        EarlyStopping(monitor="val_loss", mode="min"),
    ],
    auto_lr_find=True,
    fast_dev_run=True,
)
trainer.fit(model, dm)
