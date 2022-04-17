#!/bin/bash

# clone the repo for running evaluation
git clone https://github.com/divyanshuaggarwal/indicTrans.git
# clone requirements repositories
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git IndicTrans/indic_nlp_library
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git  IndicTrans/indic_nlp_resources
git clone https://github.com/rsennrich/subword-nmt.git IndicTrans/subword-nmt


# Install the necessary libraries
pip install -q sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
pip install -q mosestokenizer subword-nmt
# Install fairseq from source
git clone https://github.com/pytorch/fairseq.git fairseq
pip install --editable fairseq/


# # downloading the indic-en model
wget https://storage.googleapis.com/samanantar-public/V0.2/models/indic-en.zip 
unzip indic-en.zip
rm -rf indic-en.zip

# downloading the en-indic model
wget https://storage.googleapis.com/samanantar-public/V0.2/models/en-indic.zip
unzip en-indic.zip
rm -rf en-indic.zip