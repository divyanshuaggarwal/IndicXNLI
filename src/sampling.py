from datasets import load_dataset
import os
import sys
from sentence_transformers import SentenceTransformer
import copy
from numpy.random import RandomState
from dppy.finite_dpps import FiniteDPP
import numpy as np
from tqdm import tqdm
from numpy.random import rand, randn
from scipy.linalg import qr
from numpy import linalg as LA
from configs.languages import INDIC
import json
import pandas as pd
import torch

dataset_test = load_dataset("xnli", "en", split="test")
sentences = dataset_test["premise"] + dataset_test["hypothesis"]

model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
vectors = model.encode(sentences)

# Set a seed for reproducibility
eig_vecs, _ = qr(vectors)
eigenvals_sentences = LA.eigvals(eig_vecs).astype("float64")

L_rows = (eig_vecs * eigenvals_sentences).dot(eig_vecs.T)  # m x m

seed = 42
rng = np.random.RandomState(seed)

DPP = FiniteDPP(kernel_type="likelihood", L=L_rows)
k = 50
DPP.sample_exact_k_dpp(size=k, random_state=seed)

samples = DPP.list_of_samples[0]

sample_sentences = [sentences[idx] for idx in samples]

premises = dataset_test["premise"]
hypothesises = dataset_test["hypothesis"]

sents = [(premise, hypothesis) for premise, hypothesis in zip(premises, hypothesises)]

new_samples = {"sentence": [], "type": []}
for sentence in sample_sentences:
    if sentence in premises:
        for idx, premise in enumerate(premises):
            if premise == sentence and hypothesises[idx] not in new_samples["sentence"]:
                new_samples["sentence"].append(hypothesises[idx])
                new_samples["type"].append("premise")
                break
    elif sentence in hypothesises:
        for idx, premise in enumerate(hypothesises):
            if premise == sentence and premises[idx] not in new_samples["sentence"]:
                new_samples["sentence"].append(premises[idx])
                new_samples["type"].append("hypothesis")
                break

sample_sentences += [sample["sentence"] for sample in new_samples]

sys.path.append("indicTrans")
from indicTrans.inference.engine import Model

os.chdir("indicTrans")
en2indic_model = Model(expdir="../en-indic")
indic2en_model = Model(expdir="../indic-en")
os.chdir("..")

for lang in tqdm(INDIC):
    print(lang)
    with torch.no_grad():
        samples_indic_sentences = en2indic_model.batch_translate(
            sample_sentences, "en", lang
        )
        torch.cuda.empty_cache()

    sampled_data = {
        "original sentence": sample_sentences,
        f"{lang} translation": samples_indic_sentences,
        "score": [" "] * len(sample_sentences),
        "remarks": [" "] * len(sample_sentences),
        "correction": [" "] * len(sample_sentences),
    }

    df = pd.DataFrame.from_dict(sampled_data)
    df.to_csv(
        f"../sampled_data/samples_{lang}.csv",
        header=True,
        index=False,
    )
