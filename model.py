# @title Model
from argparse import Namespace

import numpy as np
import pandas as pd
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
from datasets import load_metric
import pytorch_lightning as pl
from transformers import pipeline


class TransformerForSequenceClassification(pl.LightningModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=self.hparams.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.metric = load_metric("accuracy")
        self.train_steps = self.hparams.num_steps
        self.learning_rate = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        wandb_logger = self.logger.experiment
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        labels = batch["labels"]
        scores = self.logits_to_scores(outputs.logits)

        val_step_output = {
            "val_loss": val_loss.detach().cpu(),
            "labels": labels.detach().cpu().numpy(),
            "preds": torch.argmax(logits, axis=1).detach().cpu().numpy(),
            "input_ids": batch["input_ids"].detach().cpu().numpy(),
            "y_score": scores.detach().cpu().float().numpy(),
        }
        return val_step_output

    def validation_epoch_end(self, outputs: list):
        val_loss_mean = (
            torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        )

        texts = np.concatenate(
            [
                self.tokenizer.batch_decode(x["input_ids"], skip_special_tokens=True)
                for x in outputs
            ],
            axis=0,
        )

        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        preds = np.concatenate([x["preds"] for x in outputs], axis=0)
        accuracy = self.metric.compute(predictions=preds, references=labels)

        self.log("val_loss", val_loss_mean, prog_bar=True)
        print(f"Val Loss: {val_loss_mean}, Accuracy: {accuracy}")

        self.write_outputs(texts, preds, labels)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=1e-4,
        )
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.train_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def logits_to_scores(self, logits):
        return torch.softmax(logits, dim=1)

    def write_outputs(self, texts, preds, labels):
        wandb_logger = self.logger.experiment
        pred_df = pd.DataFrame(preds, columns=["pred_label"])

        true_df = pd.DataFrame(np.array(labels), columns=["true_label"])

        text_df = pd.DataFrame({"text": texts})
        df = pd.concat((text_df, pred_df, true_df), axis=1)
        # wandb_logger.log({"table": df})
        df.to_csv(os.path.join(self.hparams.val_dir, "val_result.csv"), index=False)

    def baseline_inference(df, baseline_model, inference_column):
        inputs = df.inference_column.values.tolist()
        baseline_predictions = [baseline_model(x) for x in inputs]
        return baseline_predictions
