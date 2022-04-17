from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from datasets import Dataset
import json
import pandas as pd
from transformers import AutoTokenizer
from typing import Optional


class XNLIDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str,
        lang="en",
        batch_size=32,
        train_lang="en",
        back_translated=False,
        hypothesis_lang: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, keep_accents=True, use_fast=True
        )
        self.lang = lang
        self.batch_size = batch_size
        self.train_lang = train_lang
        self.back_translated = back_translated
        self.hypothesis_lang = hypothesis_lang
        # self.prepare_data()

    def preprocess_function(self, examples):
        # Tokenize the texts

        return self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            max_length=128,
            pad_to_max_length=True,
            truncation="longest_first",
        )

    @staticmethod
    def format_dataset(dataset):
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.model_name, keep_accents=True, use_fast=True)

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage == "train" or stage == None:
            # load train dataset
            if self.train_lang == "hi_orig":
                xnli = load_dataset("xnli", self.train_lang[:2])
                self.train_dataset = Dataset.from_dict(xnli["train"].to_dict())

            elif self.train_lang == "en":
                with open(f"data/xnli_en_train.json") as json_file:
                    data = json.load(json_file)

                train = pd.DataFrame.from_records(data["train"]).to_dict(orient="list")
                self.train_dataset = Dataset.from_dict(train)
            else:
                with open(f"data/forward/train/xnli_{self.lang}.json") as json_file:
                    data = json.load(json_file)

                train = pd.DataFrame.from_records(data["train"]).to_dict(orient="list")
                self.train_dataset = Dataset.from_dict(train)

            # load validation dataset
            if self.lang == "en" or self.lang == "hi_orig":
                xnli = load_dataset("xnli", self.lang[:2])
                self.eval_dataset = Dataset.from_dict(xnli["validation"].to_dict())

            else:
                if self.back_translated:
                    with open(f"data/backward/dev/xnli_{self.lang}.json") as json_file:
                        data = json.load(json_file)
                    val = pd.DataFrame.from_records(data["validation"]).to_dict(
                        orient="list"
                    )
                    self.eval_dataset = Dataset.from_dict(val)
                else:
                    with open(f"data/forward/dev/xnli_{self.lang}.json") as json_file:
                        data = json.load(json_file)
                    val = pd.DataFrame.from_records(data["validation"]).to_dict(
                        orient="list"
                    )
                    self.eval_dataset = Dataset.from_dict(val)

            if self.hypothesis_lang:
                if self.hypothesis_lang == "hi_orig":
                    data = load_dataset("xnli", self.lang[:2])

                    hypothesis = pd.DataFrame.from_records(data["train"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.train_dataset = self.train_dataset.to_dict()
                    self.train_dataset["hypothesis"] = hypothesis
                    self.train_dataset = Dataset.from_dict(self.train_dataset)

                    hypothesis = pd.DataFrame.from_records(data["validation"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.eval_dataset = self.eval_dataset.to_dict()
                    self.eval_dataset["hypothesis"] = hypothesis
                    self.eval_dataset = Dataset.from_dict(self.eval_dataset)
                elif self.hypothesis_lang != "en":
                    with open(
                        f"data/forward/train/xnli_{self.hypothesis_lang}.json"
                    ) as json_file:
                        data = json.load(json_file)

                    hypothesis = pd.DataFrame.from_records(data["train"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.train_dataset = self.train_dataset.to_dict()
                    self.train_dataset["hypothesis"] = hypothesis
                    self.train_dataset = Dataset.from_dict(self.train_dataset)

                    with open(
                        f"data/forward/dev/xnli_{self.hypothesis_lang}.json"
                    ) as json_file:
                        data = json.load(json_file)

                    hypothesis = pd.DataFrame.from_records(data["validation"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.eval_dataset = self.eval_dataset.to_dict()
                    self.eval_dataset["hypothesis"] = hypothesis
                    self.eval_dataset = Dataset.from_dict(self.eval_dataset)

                else:
                    with open(f"data/xnli_en_train.json") as json_file:
                        data = json.load(json_file)

                    hypothesis = pd.DataFrame.from_records(data["train"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.train_dataset = self.train_dataset.to_dict()
                    self.train_dataset["hypothesis"] = hypothesis
                    self.train_dataset = Dataset.from_dict(self.train_dataset)

                    data = load_dataset("xnli", "en")

                    hypothesis = pd.DataFrame.from_records(data["validation"]).to_dict(
                        orient="list"
                    )["hypothesis"]
                    self.eval_dataset = self.eval_dataset.to_dict()
                    self.eval_dataset["hypothesis"] = hypothesis
                    self.eval_dataset = Dataset.from_dict(self.eval_dataset)

            self.train_dataset = self.train_dataset.map(
                self.preprocess_function,
                batched=True,
                # num_proc = 10,
                desc="Running tokenizer on train dataset",
            )

            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                # num_proc = 10,
                desc="Running tokenizer on val dataset",
            )

            self.train_dataset.rename_column_("label", "labels")
            self.eval_dataset.rename_column_("label", "labels")

            self.train_dataset = self.format_dataset(self.train_dataset)
            self.eval_dataset = self.format_dataset(self.eval_dataset)

        if stage == "test" or stage == None:
            if self.lang == "en" or self.lang == "hi_orig":
                xnli = load_dataset("xnli", self.lang[:2])
                self.test_dataset = Dataset.from_dict(xnli["test"].to_dict())
            else:
                if self.back_translated:
                    with open(f"data/backward/test/xnli_{self.lang}.json") as json_file:
                        data = json.load(json_file)
                    test = pd.DataFrame.from_records(data["test"]).to_dict(
                        orient="list"
                    )
                    self.test_dataset = Dataset.from_dict(test)
                else:
                    with open(f"data/forward/test/xnli_{self.lang}.json") as json_file:
                        data = json.load(json_file)
                    test = pd.DataFrame.from_records(data["test"]).to_dict(
                        orient="list"
                    )
                    self.test_dataset = Dataset.from_dict(test)

            if self.hypothesis_lang:
                data = {}
                if self.hypothesis_lang in ("en", "hi_orig"):
                    data = load_dataset("xnli", self.lang[:2])
                else:
                    with open(
                        f"data/forward/test/xnli_{self.hypothesis_lang}.json"
                    ) as json_file:
                        data = json.load(json_file)

                hypothesis = pd.DataFrame.from_records(data["test"]).to_dict(
                    orient="list"
                )["hypothesis"]
                self.test_dataset = self.test_dataset.to_dict()
                self.test_dataset["hypothesis"] = hypothesis
                self.test_dataset = Dataset.from_dict(self.test_dataset)

            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                # num_proc = 10,
                desc="Running tokenizer on val dataset",
            )
            self.test_dataset.rename_column_("label", "labels")
            self.test_dataset = self.format_dataset(self.test_dataset)

    # return the dataloader for each split
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler = sampler
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            # sampler = sampler
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )
        return test_dataloader
