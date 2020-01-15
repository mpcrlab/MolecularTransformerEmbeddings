# MolecularTransformerEmbeddings
Code for the Transformer neural network trained to translate between molecular text representations and create molecular embeddings.

Network weights of a pretrained Transformer trained to translate between 83,000,000+ SMILES/IUPAC string pairs collected from PubChem are available for download, along with code to obtain embeddings for a list of SMILES strings.

The Encoder of the Transformer network can convert a SMILES string of length M into an MxN matrix, where N is the size of the embedding vectors used for each SMILES string character.

![Plot of 20 amino acids in dimensionality-reduced embedding space, using pretrained Transformer embeddings.](https://github.com/mpcrlab/MolecularTransformerEmbeddings/blob/images/aa_emb.png)

All testing was done on an Ubuntu 16.04 using Python 3.6.8 with NVIDIA GPUs. No GPU resources are required, but may be useful to speed up training/finetuning.

## Installation

1. Ensure Python 3 is installed.
2. Git clone this repository using the following command:
```
git clone https://github.com/mpcrlab/MolecularTransformerEmbeddings.git
```
3. (Optional) To use GPU resources, make sure the most recent NVIDIA driver is installed on your system.
4. In the cloned repository, install required python libraries with this command:
```
pip3 install -r requirements.txt
```
5. Download string pairs and pretrained weights:
```
chmod +x download.sh
./download.sh
```
6. (Optional) To use jupyter notebooks, start a notebook server:
```
jupyter notebook
```

## Obtaining Embeddings

## Training or Finetuning the Transformer Network
