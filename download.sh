#!/bin/bash

mkdir -p data
mkdir -p checkpoints

wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_1m.tsv -O data/smiles_iupac_train_1m.tsv
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/checkpoints/pretrained.ckpt -O checkpoints/pretrained.ckpt
