import argparse
from argparse import Namespace
import os
from datetime import datetime
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from utility import load_train_data
from loader import Encoder
from loader import MultiClassDataModule
from model import TransformerForSequenceClassification


def main(hparams) -> None:
    pl.seed_everything(seed=42)

    train_df, val_df = load_train_data(hparams)
    # Initialize MultiClassDataModule
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name_or_path)
    encoder = Encoder(tokenizer)
    dm = MultiClassDataModule(train_df, val_df, tokenizer, encoder)
    dm.setup()
    hparams.num_steps = hparams.max_epochs * len(dm.train_dataloader())

    # Initialize TransformerForSequenceClassification
    model = TransformerForSequenceClassification(hparams)

    # Create Logger and Callbacks
    wandb_logger = WandbLogger(project="grid-solvvy")
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor, verbose=True, mode="min"
    )
    ckpt_path = os.path.join(
        "./experiments/",
        "version_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "checkpoints",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        mode="min",
    )

    # Initialize Trainer and Fit
    trainer = pl.Trainer(
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator=hparams.accelerator,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    args = Namespace(
        model_name_or_path="distilbert-base-uncased",
        num_labels=3,
        save_top_k=1,
        max_epochs=1,
        batch_size=16,
        accelerator="auto",
        data_dir="/Users/james.morrissey/Downloads/solvvy_one_shot_data",
        # data_name="zero_shot_labels_1000_2000.csv",
        fast_dev_run=True,
        monitor="val_loss",
    )

    # parser = argparse.ArgumentParser(
    #     description="Train a Transformer model for sequence classification",
    #     add_help=True,
    # )
    # parser.add_argument("--fast_dev_run", type=bool, default=False)
    # parser.add_argument("--model_name_or_path", type=str)
    # parser.add_argument("--num_labels", type=int)

    # parser.add_argument(
    #     "--save_top_k",
    #     default=1,
    #     type=int,
    #     help="The best k models will be saved according to the quantity monitored.",
    # )
    # # Early Stopping
    # parser.add_argument(
    #     "--monitor", default="val_loss", type=str, help="Quantity to monitor."
    # )
    # parser.add_argument(
    #     "--max_epochs",
    #     default=3,
    #     type=int,
    #     help="Stop training after this number of epochs.",
    # )
    # parser.add_argument("--batch_size", default=16, type=int)
    # parser.add_argument("--accelerator", default="auto", type=str)
    # parser.add_argument("--accelerator", default="auto", type=str)
    # args = parser.parse_args()

    main(hparams=args)
