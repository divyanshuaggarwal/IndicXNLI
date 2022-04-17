import os
import sys
from datasets import load_dataset
import json
from tqdm import tqdm
from configs.languages import INDIC
from configs.models import big_models, small_models

xnli = load_dataset("xnli", "en")


def translate_dataset(dataset, src="en", dest="hi"):
    train_premise = [item["premise"] for item in dataset["train"]]
    test_premise = [item["premise"] for item in dataset["test"]]
    val_premise = [item["premise"] for item in dataset["validation"]]

    train_hypothesis = [item["hypothesis"] for item in dataset["train"]]
    test_hypothesis = [item["hypothesis"] for item in dataset["test"]]
    val_hypothesis = [item["hypothesis"] for item in dataset["validation"]]

    if src == "en":
        train_premise = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(train_premise)
        ]
        test_premise = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(test_premise)
        ]
        val_premise = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(val_premise)
        ]

        train_hypothesis = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(train_hypothesis)
        ]
        test_hypothesis = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(test_hypothesis)
        ]
        val_hypothesis = [
            en2indic_model.translate_paragraph(item, "en", dest)
            for item in tqdm(val_hypothesis)
        ]

    else:
        train_premise = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(train_premise)
        ]
        test_premise = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(test_premise)
        ]
        val_premise = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(val_premise)
        ]

        train_hypothesis = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(train_hypothesis)
        ]
        test_hypothesis = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(test_hypothesis)
        ]
        val_hypothesis = [
            indic2en_model.translate_paragraph(item, dest, "en")
            for item in tqdm(val_hypothesis)
        ]

    train_data = []
    test_data = []
    val_data = []

    for i in range(len(train_premise)):
        new_item = {
            "premise": train_premise[i],
            "hypothesis": train_hypothesis[i],
            "label": dataset["train"][i]["label"],
        }

        train_data.append(new_item)

    for i in range(len(test_premise)):
        new_item = {
            "premise": test_premise[i],
            "hypothesis": test_hypothesis[i],
            "label": dataset["test"][i]["label"],
        }

        test_data.append(new_item)

    for i in range(len(val_premise)):
        new_item = {
            "premise": val_premise[i],
            "hypothesis": val_hypothesis[i],
            "label": dataset["validation"][i]["label"],
        }

        val_data.append(new_item)

    new_dataset = {"train": train_data, "test": test_data, "validation": val_data}

    return new_dataset


if __name__ == "__main__":
    sys.path.append("IndicTrans")
    from indicTrans.inference.engine import Model

    os.chdir("indicTrans")
    en2indic_model = Model(expdir="../en-indic")
    indic2en_model = Model(expdir="../indic-en")

    # forward translate
    for lang in INDIC:
        new_dataset = translate_dataset(xnli, "en", lang)

        out_file = open(f"../data/forward/train/xnli_{lang}.json", "w", encoding="utf8")
        json.dump(new_dataset["train"], out_file, indent=6, ensure_ascii=False)
        out_file.close()

        out_file = open(f"../data/forward/test/xnli_{lang}.json", "w", encoding="utf8")
        json.dump(new_dataset["test"], out_file, indent=6, ensure_ascii=False)
        out_file.close()

        out_file = open(f"../data/forward/dev/xnli_{lang}.json", "w", encoding="utf8")
        json.dump(new_dataset["validation"], out_file, indent=6, ensure_ascii=False)
        out_file.close()

    # backward translate
    for lang in INDIC:
        data = {}
        with open(f"../data/forward/train/xnli_{lang}.json") as f:
            data["train"] = json.load(f)

        with open(f"../data/forward/dev/xnli_{lang}.json") as f:
            data["validation"] = json.load(f)

        with open(f"../data/forward/test/xnli_{lang}.json") as f:
            data["test"] = json.load(f)

        new_dataset = translate_dataset(data, lang, "en")

        out_file = open(
            f"../data/backward/train/xnli_{lang}.json", "w", encoding="utf8"
        )
        json.dump(new_dataset["train"], out_file, indent=6, ensure_ascii=False)
        out_file.close()

        out_file = open(f"../data/backward/test/xnli_{lang}.json", "w", encoding="utf8")
        json.dump(new_dataset["test"], out_file, indent=6, ensure_ascii=False)
        out_file.close()

        out_file = open(f"../data/backward/dev/xnli_{lang}.json", "w", encoding="utf8")
        json.dump(new_dataset["validation"], out_file, indent=6, ensure_ascii=False)
        out_file.close()
