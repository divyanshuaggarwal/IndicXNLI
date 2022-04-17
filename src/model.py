import os
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
)
import json
import torch
from datasets import load_metric
from pytorch_lightning import LightningModule
from typing import Optional


class XNLIModel(LightningModule):
    def __init__(
        self,
        model_name: str,
        #  config:Optional[str] = None,
        lang: Optional[str] = None,
        learning_rate=2e-5,
        batch_size=32,
        weight_decay=0.01,
        eps=1e-6,
        warmup_steps=150,
        suffix="en",
    ):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
        )
        self.lang = lang
        self.batch_size = batch_size
        self.metric = load_metric("xnli")
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.suffix = suffix

    def forward(self, **x):
        outputs = self.model(**x)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        # self.log("val_loss",val_loss)
        return {"val_loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"test_loss": test_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = self.metric.compute(predictions=preds, references=labels)["accuracy"]
        # self.log_dict(self.metric.compute(predictions=preds, references=labels))
        self.log("val_loss", val_loss)
        tqdm_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        self.log("val_acc", val_acc)
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log_dict(self.metric.compute(predictions=preds, references=labels))
        results = {}
        results[self.lang] = self.metric.compute(predictions=preds, references=labels)[
            "accuracy"
        ]
        save_path = self.model_name

        if "/" in self.model_name:
            save_path = self.model_name.split("/")[-1]

        if f"xnli_experiment_{save_path}_{self.suffix}.json" in os.listdir("results/"):
            with open(
                f"results/xnli_experiment_{save_path}_{self.suffix}.json",
                "r",
            ) as f:
                old_results = json.load(f)
            old_results[self.lang] = results[self.lang]
            results = old_results

        with open(
            f"results/xnli_experiment_{save_path}_{self.suffix}.json",
            "w",
        ) as f:
            json.dump(results, f, indent=6)

        tqdm_dict = {
            "test_loss": loss,
            "test_acc": results[self.lang],
        }

        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "test_loss": loss}

    def prepare_data(self):
        AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )
        load_metric("xnli")

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)
        return optimizer
