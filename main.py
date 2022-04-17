import json
import os
from src.datamodule import XNLIDataModule
from src.model import XNLIModel
from src.trainer import get_trainer
from configs.languages import INDIC
from configs.models import small_models, big_models
import torch

models = small_models + big_models


def indic_train():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_in.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_in.json", "r") as f:
            results = json.load(f)

    for lang in INDIC:
        torch.cuda.empty_cache()
        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        BATCH_SIZE = 128 if model_name not in big_models else 64
        # BATCH_SIZE = 128
        SUFFIX = "in"
        print(f"using {model_name}")
        print(f"training on {lang}")
        model = XNLIModel(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, suffix=SUFFIX
        )
        dm = XNLIDataModule(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, train_lang=lang
        )
        # trainer.tune(model,dm)
        trainer.fit(model, dm)
        try:
            trainer.test(model, dm)
        except:
            pass


def english_train():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_en.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_en.json", "r") as f:
            results = json.load(f)

    for lang in INDIC:
        # torch.cuda.empty_cache()
        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        SUFFIX = "en"
        print(f"using {model_name}")
        print(f"training on {lang}")
        BATCH_SIZE = 128 if model_name not in big_models else 64
        # BATCH_SIZE = 128
        model = XNLIModel(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
            suffix=SUFFIX,
            learning_rate=2e-5,
        )
        dm = XNLIDataModule(model_name=model_name, lang=lang, batch_size=BATCH_SIZE)

        trainer.fit(model, dm)
        try:
            trainer.test(model, dm)
        except:
            pass


def english_indic_train():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_2_step.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_2_step.json", "r") as f:
            results = json.load(f)

    for lang in INDIC:

        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        BATCH_SIZE = 128 if model_name not in big_models else 64
        SUFFIX = "2_step"
        print(f"using {model_name}")
        print(f"training on {lang}")
        dm = XNLIDataModule(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
        )
        dm_2 = XNLIDataModule(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, train_lang=lang
        )
        model = XNLIModel(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, suffix=SUFFIX
        )
        trainer.fit(model, dm)
        del trainer
        trainer = get_trainer()
        trainer.fit(model,dm_2)
        
        try:
            trainer.test(model, dm_2)
        except:
            pass
        del trainer


def english_eval():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_back.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_back.json", "r") as f:
            results = json.load(f)

    for lang in INDIC:
        torch.cuda.empty_cache()
        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        SUFFIX = "back"
        print(f"using {model_name}")
        print(f"training on {lang}")
        BATCH_SIZE = 128 if model_name not in big_models else 64
        # BATCH_SIZE = 128
        model = XNLIModel(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
            suffix=SUFFIX,
        )
        dm = XNLIDataModule(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
            back_translated=True,
        )

        trainer.fit(model, dm)
        try:
            trainer.test(model, dm)
        except:
            pass


def train_all():
    for model_name in models:
        print(model_name)
        SUFFIX = "n_step"
        BATCH_SIZE = 128 if model_name not in big_models else 64

        save_path = model_name

        if "/" in model_name:
            save_path = model_name.split("/")[-1]

        if f"xnli_experiment_{save_path}_n_step.json" in os.listdir("results/"):
            with open(f"results/xnli_experiment_{save_path}_n_step.json", "r") as f:
                results = json.load(f)
            if len(results) == len(INDIC):
                print(f"skipping {model_name}")
                continue

        trainer = get_trainer()

        model = XNLIModel(model_name=model_name, batch_size=BATCH_SIZE, suffix=SUFFIX)

        for lang in list(["en"] + INDIC[:-1]):

            torch.cuda.empty_cache()

            dm = XNLIDataModule(
                model_name=model_name, lang=lang, batch_size=BATCH_SIZE, train_lang=lang
            )

            print(f"using {model_name}")
            print(f"training on {lang}")

            trainer.fit(model, dm)

            del trainer
            trainer = get_trainer()

        for test_lang in INDIC:
            model.lang = test_lang
            test_dm = XNLIDataModule(
                model_name=model_name, lang=test_lang, batch_size=BATCH_SIZE
            )
            test_dm.setup("test")
            try:
                trainer.test(model, test_dm.test_dataloader())
            except:
                pass
        try:
            trainer.save_checkpoint(
                f"results/pretrained_models/{save_path}_n_step.ckpt"
            )
        except:
            pass


def cross_lingual_transfer():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_in.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_in.json", "r") as f:
            results = json.load(f)

    for lang in list(["en"] + INDIC):

        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        BATCH_SIZE = 128 if model_name not in big_models else 64

        SUFFIX = f"{lang}_in"
        print(f"using {model_name}")
        print(f"training on {lang}")
        model = XNLIModel(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, suffix=SUFFIX
        )
        dm = XNLIDataModule(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, train_lang=lang
        )
        # trainer.tune(model,dm)
        trainer.fit(model, dm)
        for test_lang in INDIC:
            model.lang = test_lang
            dm_2 = XNLIDataModule(
                model_name=model_name,
                lang=test_lang,
                batch_size=BATCH_SIZE,
                train_lang=test_lang,
            )
            try:
                trainer.test(model, dm_2)
            except:
                pass


def enindic_english_indic_train():
    for model_name in models:
        print(model_name)
    results = {}
    save_path = model_name

    if "/" in model_name:
        save_path = model_name.split("/")[-1]

    if f"xnli_experiment_{save_path}_hypo_2_step.json" in os.listdir("results/"):
        with open(f"results/xnli_experiment_{save_path}_hypo_2_step.json", "r") as f:
            results = json.load(f)

    for lang in INDIC:

        if lang in results.keys():
            print(f"skipping {model_name} for language {lang}")
            continue

        trainer = get_trainer()

        BATCH_SIZE = 128 if model_name not in big_models else 64
        SUFFIX = "hypo_2_step"
        print(f"using {model_name}")
        print(f"training on {lang}")
        dm = XNLIDataModule(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
        )
        dm_2 = XNLIDataModule(
            model_name=model_name,
            lang=lang,
            batch_size=BATCH_SIZE,
            train_lang=lang,
            hypothesis_lang=lang,
        )
        model = XNLIModel(
            model_name=model_name, lang=lang, batch_size=BATCH_SIZE, suffix=SUFFIX
        )

        trainer.fit(model, dm)

        trainer = get_trainer()

        trainer.fit(model, dm_2)

        try:
            trainer.test(model, dm_2)
        except:
            pass
        del trainer


def enindic_train_all():
    for model_name in models:
        print(model_name)
        SUFFIX = "hypo_n_step"
        BATCH_SIZE = 128 if model_name not in big_models else 64

        save_path = model_name

        if "/" in model_name:
            save_path = model_name.split("/")[-1]

        if f"xnli_experiment_{save_path}_hypo_n_step.json" in os.listdir("results/"):
            print(f"skipping {model_name}")
            continue

        trainer = get_trainer()

        model = XNLIModel(model_name=model_name, batch_size=BATCH_SIZE, suffix=SUFFIX)

        for lang in list(["en"] + INDIC):

            dm = XNLIDataModule(
                model_name=model_name,
                lang=lang,
                batch_size=BATCH_SIZE,
                train_lang="en",
                hypothesis_lang=lang,
            )

            print(f"using {model_name}")
            print(f"training on {lang}")

            trainer.fit(model, dm)

            del trainer
            trainer = get_trainer()

        for test_lang in INDIC:
            model.lang = test_lang
            test_dm = XNLIDataModule(
                model_name=model_name,
                lang=test_lang,
                batch_size=BATCH_SIZE,
                hypothesis_lang=test_lang,
            )
            test_dm.setup("test")
            try:
                trainer.test(model, test_dm.test_dataloader())
            except:
                pass


if __name__ == "__main__":
    indic_train()
    english_train()
    english_indic_train()
    english_eval()
    train_all()
    cross_lingual_transfer()
    enindic_english_indic_train()
    enindic_train_all()
