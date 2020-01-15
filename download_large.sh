#!/bin/bash

mkdir -p data

wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.aa -O data/smiles_iupac_train_83m.tsv.aa
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.ab -O data/smiles_iupac_train_83m.tsv.ab
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.ac -O data/smiles_iupac_train_83m.tsv.ac
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.ad -O data/smiles_iupac_train_83m.tsv.ad
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.ae -O data/smiles_iupac_train_83m.tsv.ae
wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_83m.tsv.af -O data/smiles_iupac_train_83m.tsv.af

cat data/smiles_iupac_train_83m.tsv.aa data/smiles_iupac_train_83m.tsv.ab data/smiles_iupac_train_83m.tsv.ac data/smiles_iupac_train_83m.tsv.ad data/smiles_iupac_train_83m.tsv.ae data/smiles_iupac_train_83m.tsv.af > data/smiles_iupac_train_83m.tsv

sudo rm -rf data/smiles_iupac_train_83m.tsv.a*
