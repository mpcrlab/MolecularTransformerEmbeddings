# MolecularTransformerEmbeddings
Code for the Transformer neural network trained to translate between molecular text representations and create molecular embeddings.

The Encoder of the Transformer network can convert a SMILES string of length M into an MxN matrix, where N is the size of the embedding vectors used for each SMILES string character. Pretrained weights from a network trained on 83,000,000+ molecules are provided so useful embeddings can be obtained.

![Plot of 20 amino acids in dimensionality-reduced embedding space, using pretrained Transformer embeddings.](https://github.com/mpcrlab/MolecularTransformerEmbeddings/blob/images/aa_emb.png)

All testing was done on an Ubuntu 16.04 using Python 3.6.8 with NVIDIA GPUs. No GPU resources are required, but may be useful to speed up training/finetuning.

## Installation

1. Ensure Python 3 is installed.
2. Git clone this repository using the following command:
'''
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
'''

## Obtaining Embeddings

## Training or Finetuning the Transformer Network
