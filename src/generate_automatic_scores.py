import pandas as pd
import json
import os
from collections import defaultdict
from configs.languages import INDIC

from googletrans import Translator
from tqdm import tqdm

from datasets import load_metric, load_dataset


def get_translations(sentences, src, dest):
    res = []

    for sentence in tqdm(sentences, desc="translating"):
        translator = Translator()
        res.append(translator.translate(sentence, src=src, dest=dest).text)

    return res


def open_file(path, split="test"):
    with open(path, "r") as f:
        dic = json.load(f)

    records = dic[split]

    return pd.DataFrame.from_records(records)


bertscore = load_metric("bertscore")

for lang in INDIC:
    print(lang)

    xnli = load_dataset("xnli", "en", split="test")
    sample_sentences = xnli["premise"] + xnli["hypothesis"]
    df = open_file(f"data/forward/test/xnli_{lang}.json")
    samples_indic_sentences = df["premise"].to_list() + df["hypothesis"].to_list()
    df = open_file(f"data/backward/test/xnli_{lang}.json")
    samples_indic_sentences_back = df["premise"].to_list() + df["hypothesis"].to_list()

    google_trans_forward = []
    google_trans_backward = []
    google_trans_indic_sent_back = [" " for _ in sample_sentences]

    sampled_data = defaultdict(list)

    if lang != "as":
        google_trans_forward = get_translations(sample_sentences, src="en", dest=lang)
        google_trans_backward = get_translations(
            google_trans_forward, src=lang, dest="en"
        )

    sampled_data["original sentence"] = sample_sentences
    sampled_data[f"{lang} translation"] = samples_indic_sentences
    sampled_data["back translated"] = samples_indic_sentences_back

    print(len(sampled_data["original sentence"]))
    print(len(sampled_data[f"{lang} translation"]))
    print(len(sampled_data["back translated"]))

    sampled_data[
        f"google translation of sentence from en to {lang}"
    ] += google_trans_forward
    sampled_data[
        f"google translation of sentence from {lang} to en (back translation)"
    ] += google_trans_backward

    sampled_data["bertscore f1 original vs back translated"] += bertscore.compute(
        predictions=sample_sentences, references=samples_indic_sentences_back, lang="en"
    )["f1"]
    sampled_data[
        "bertscore f1 original vs back translated googletrans"
    ] += bertscore.compute(
        predictions=sample_sentences, references=google_trans_backward, lang="en"
    )[
        "f1"
    ]

    sampled_data[
        "bertscore precision original vs back translated"
    ] += bertscore.compute(
        predictions=sample_sentences, references=samples_indic_sentences_back, lang="en"
    )[
        "precision"
    ]
    sampled_data[
        "bertscore precision original vs back translated googletrans"
    ] += bertscore.compute(
        predictions=sample_sentences, references=google_trans_backward, lang="en"
    )[
        "precision"
    ]

    sampled_data["bertscore recall original vs back translated"] += bertscore.compute(
        predictions=sample_sentences, references=samples_indic_sentences_back, lang="en"
    )["recall"]
    sampled_data[
        "bertscore recall original vs back translated googletrans"
    ] += bertscore.compute(
        predictions=sample_sentences, references=google_trans_backward, lang="en"
    )[
        "recall"
    ]

    res = {}
    for k, v in list(sampled_data.items())[5:]:
        res[k] = sum(v) / len(v)

    print(res)

    with open(f"automatic_scores/test_set_{lang}_avg.json", "w") as f:
        json.dump(res, f)
    df = pd.DataFrame(sampled_data)
    df.to_csv(f"automatic_scores/bert_score_{lang}.csv", index=False)
