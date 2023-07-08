# IndicXNLI

Implementation of the translation pipeline, automatic sampling and scoring,human evaluation and experiments in our EMNLP 2022 paper: [IndicXNLI: Evaluating Multilingual Inference for Indian Languages](https://aclanthology.org/2022.emnlp-main.755/). To explore the dataset online visit [dataset page](https://huggingface.co/datasets/Divyanshu/indicxnli).
```
@inproceedings{aggarwal-etal-2022-indicxnli,
    title = "{I}ndic{XNLI}: Evaluating Multilingual Inference for {I}ndian Languages",
    author = "Aggarwal, Divyanshu  and
      Gupta, Vivek  and
      Kunchukuttan, Anoop",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.755",
    pages = "10994--11006",
    abstract = "While Indic NLP has made rapid advances recently in terms of the availability of corpora and pre-trained models, benchmark datasets on standard NLU tasks are limited. To this end, we introduce INDICXNLI, an NLI dataset for 11 Indic languages. It has been created by high-quality machine translation of the original English XNLI dataset and our analysis attests to the quality of INDICXNLI. By finetuning different pre-trained LMs on this INDICXNLI, we analyze various cross-lingual transfer techniques with respect to the impact of the choice of language models, languages, multi-linguality, mix-language input, etc. These experiments provide us with useful insights into the behaviour of pre-trained models for a diverse set of languages.",
}
```

Below are the details about the [IndicXNLI datasets](https://huggingface.co/datasets/Divyanshu/indicxnli) and scripts for reproducing the results reported in the EMNLP 2022 Paper.

## 0. Prerequisites
The code requires `python 3.7+` 

Clone this repository on your machine - `git clone https://github.com/divyanshuaggarwal/IndicXNLI.git` 

create a conda environment using ```conda create --name indicxnli python=3.7``` command and activate it using   ```conda activate indicxnli```

Install requirements by typing the following command-
```pip install -r requirements.txt``` 

Download and unpack the [IndicXNLI dataset](https://huggingface.co/datasets/Divyanshu/indicxnli/tree/main) into ```./data``` in the main ```IndicXNLI``` folder. 

Carefully read the <b>LICENCE</b> for non-academic usage. 

### 0.1 Downloading the Dataset
run the following the commands from the root of the directory
```
cd data
git lfs install
git clone https://huggingface.co/datasets/Divyanshu/indicxnli
```

After downloading, you'll have multiple sub-folders with several json files. Each json file in the sub-folders is a list of json objects with premise, hypothesis and label key. The folder structure will be as follows:


```
data
├── backward
│   ├── dev
│   │   ├── xnli_as.json
│   │   ├── xnli_bn.json
│   │   ├── xnli_gu.json
│   │   ├── xnli_hi.json
│   │   ├── xnli_kn.json
│   │   ├── xnli_ml.json
│   │   ├── xnli_mr.json
│   │   ├── xnli_or.json
│   │   ├── xnli_pa.json
│   │   ├── xnli_ta.json
│   │   └── xnli_te.json
│   └── test
│       ├── xnli_as.json
│       ├── xnli_bn.json
│       ├── xnli_gu.json
│       ├── xnli_hi.json
│       ├── xnli_kn.json
│       ├── xnli_ml.json
│       ├── xnli_mr.json
│       ├── xnli_or.json
│       ├── xnli_pa.json
│       ├── xnli_ta.json
│       └── xnli_te.json
└── forward
    ├── dev
    │   ├── xnli_as.json
    │   ├── xnli_bn.json
    │   ├── xnli_gu.json
    │   ├── xnli_hi.json
    │   ├── xnli_kn.json
    │   ├── xnli_ml.json
    │   ├── xnli_mr.json
    │   ├── xnli_or.json
    │   ├── xnli_pa.json
    │   ├── xnli_ta.json
    │   └── xnli_te.json
    ├── test
    │   ├── xnli_as.json
    │   ├── xnli_bn.json
    │   ├── xnli_gu.json
    │   ├── xnli_hi.json
    │   ├── xnli_kn.json
    │   ├── xnli_ml.json
    │   ├── xnli_mr.json
    │   ├── xnli_or.json
    │   ├── xnli_pa.json
    │   ├── xnli_ta.json
    │   └── xnli_te.json
    └── train
        ├── xnli_as.json
        ├── xnli_bn.json
        ├── xnli_gu.json
        ├── xnli_hi.json
        ├── xnli_kn.json
        ├── xnli_ml.json
        ├── xnli_mr.json
        ├── xnli_or.json
        ├── xnli_pa.json
        ├── xnli_ta.json
        └── xnli_te.json

7 directories, 55 files
```
## 1.Data Set
```data/forward```  and ```data/backward``` will be the primary datasets folders to work on here. ```data/forward``` folder contains `en-->in` translations of premise and hypothesis and `data/backward` folder contain `in-->en` back translations from the `data/forward` folder.
### 1.1 Translation
The original english XNLI Dataset is translated using the IndicTrans translation model from AI4Bharat.

To install IndicTrans run ``` bash setup_indic_trans.sh``` in the terminal.

You can then start creating the dataset by running ```python src/make_dataset.py```.This will create the dataset in the data folder by the above mentioned tree format.

<b>Note:</b> Before running any python script, first run `export PYTHONPATH=$PWD` in the terminal.

### 1.2 Sampling Using DPP
To sample out the examples fron test dataset using dpp run ```python src/sampling.py``` in the terminal. This will create the necessary files in the `./sampled_data` directory:

```
sampled_data
├── samples_as.csv
├── samples_gu.csv
├── samples_or.csv
├── samples_kn.csv
├── samples_bn.csv
├── samples_hi.csv
├── samples_pa.csv
├── samples_mr.csv
├── samples_ml.csv
├── samples_ta.csv
└── samples_te.csv
```

These datasets are then copied to ```./human_evaluation_data/``` for human annotators to evaluate the translation quality.
### 1.3 Bert Score on Test Data
We calculate BertScore on the original test data and backtranslated test data. This allows us to identify the upper bound error. to run the python scripts run ```python src/generate_automatic_scores.py``` which will populate the ```./automatic_scores/``` folder in the format ```test_set_{language code}_avg.json``` like mentioned below: 

```
automatic_score
├── test_set_as_avg.json
├── test_set_gu_avg.json
├── test_set_kn_avg.json
├── test_set_pa_avg.json
├── test_set_mr_avg.json
├── test_set_ml_avg.json
├── test_set_hi_avg.json
├── test_set_bn_avg.json
├── test_set_ta_avg.json
├── test_set_te_avg.json
└── test_set_or_avg.json
```

The json file will of format, where key is the bertscore of original vs the indic trans or google trans and value is the average bertscore across all test samples.For examples refer below:

```json
    {
    "bertscore f1 original vs back translated": 0.9366865618024282, 
    "bertscore f1 original vs back translated googletrans": 0.9743754622941008,
    "bertscore precision original vs back translated": 0.931326547413767,
    "bertscore precision original vs back translated googletrans": 0.9859856570433237, 
    "bertscore recall original vs back translated": 0.9423049956560134,
    "bertscore recall original vs back translated googletrans": 0.9659530529480971,
    }
```

The script will also generate sentence-wise BertScore ```.csv``` file for every language as well in the ```./automatic_scores``` folder.

## 2. Baselines
The Expirements were run on google colab TPUs. To setup TPU environment run the following in the terminal:
```
bash setup_tpu.sh
```
also uncomment <b>line 17</b> in ```src/trainer.py``` file.

To run the baselines type the following in the terminal: 
```
python main.py
```
this will create the result json files in the `./results/` folder.

a typical result json file looks like following:
```xnli_experiments_{model name}_{baseline suffix code}.json ```

in a typical json file the data is organised in key value pairs where key is the language code and the value is the accuracy on that language for example:
```json
    {
    "as": 0.72,
    "gu": 0.85,
    "kn": 0.75,
    ....
    }
```
the baselines are named as follows:

|Baseline as Named in Paper|Baselines suffix codes for JSON file|
|:----|:----|
|Indic Train|in|
|English Train|en|
|Eglish Eval|back|
|English + Indic Train|2_step|
|Train all|n_step|
|Crosslingual Transfer|{language code for which the model is trained on for ex. as,gu etc}_in|
|<b>En-indic</b>| <b> - </b> |
| English + Indic Train|2_step_hypo|
|Train all|n_step_hypo|
