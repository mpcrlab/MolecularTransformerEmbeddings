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

1. To obtain embeddings for SMILES strings, simply create a text file with one string per line. Save this file in the data folder as ```YOUR_SMILES_STRINGS.txt```. An example file with 20 amino acids is included:
```
C(CC(C(=O)O)N)CN=C(N)N
C1=C(NC=N1)CC(C(=O)O)N
CCC(C)C(C(=O)O)N
CC(C)CC(C(=O)O)N
C(CCN)CC(C(=O)O)N
CSCCC(C(=O)O)N
C1=CC=C(C=C1)CC(C(=O)O)N
CC(C(C(=O)O)N)O
C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N
CC(C)C(C(=O)O)N
CC(C(=O)O)N
C(C(C(=O)O)N)C(=O)N
C(C(C(=O)O)N)C(=O)O
C(CC(=O)O)C(C(=O)O)N
C(C(=O)O)N
C(C(C(=O)O)N)O
C1=CC(=CC=C1CC(C(=O)O)N)O
C(C(C(=O)O)N)S
C(CC(=O)N)C(C(=O)O)N
C1CC(NC1)C(=O)O
```
2. In the cloned repository run the following command to generate embeddings for your custom file:
```
python3 embed.py --data_path=data/YOUR_SMILES_STRINGS.txt
```
3. Embeddings are saved as a zip of numpy arrays, where each numpy array is an MxN matrix for a molecule. The arrays can be loaded in Python like this:
```
import numpy as np

arrays = np.load("embeddings/YOUR_SMILES_STRINGS.npz")
print(arrays['C1CC(NC1)C(=O)O'])
```
4. (Optional) use the ```view_embeddings.ipynb``` jupyter notebook to use embeddings interactively instead of saving them in the embeddings folder.

(Note: Pretrained weights must be downloaded using ```download.sh``` before embeddings may be obtained. Custom weights may be supplied using the ```--checkpoint_path``` flag when calling ```embed.py```)

## Training or Finetuning the Transformer Network

* Code for training the Transformer network on a SMILES-IUPAC string translation task is provided in ```train.py```. Pretrained weights are located in ```checkpoints/pretrained.ckpt```.

* Training data is stored in Tab-Separated Value files in the ```data``` folder. The files do not have any header row. Each line contains a SMILES string followed by a TAB character, an IUPAC name, and a `\n` newline character. Example data is provided, and custom training data can be used by creating a file ```data/YOUR_TRAINING_DATA.tsv``` with this structure:
```
CC1(CCC2C1CN(C2C(=O)NC(CC3CCC3)C(=O)C(=O)NCC=C)C(=O)C(C4CCCCC4)NC(=O)NC(CN5C(=O)CC(CC5=O)(C)C)C(C)(C)C)C 	(3S,3aS,6aS)-N-[3-(allylamino)-1-(cyclobutylmethyl)-2,3-dioxo-propyl]-2-[(2S)-2-cyclohexyl-2-[[(1S)-1-[(4,4-dimethyl-2,6-dioxo-1-piperidyl)methyl]-2,2-dimethyl-propyl]carbamoylamino]acetyl]-6,6-dimethyl-1,3,3a,4,5,6a-hexahydrocyclopenta[c]pyrrole-3-carboxamide
C1=CN(C(=O)NC1=O)C2C(C(C(O2)C(OP(=O)(O)OP(=O)(O)OP(=O)(O)O)F)O)O 	[[(S)-[(2S,3S,4R,5R)-5-(2,4-dioxopyrimidin-1-yl)-3,4-dihydroxy-tetrahydrofuran-2-yl]-fluoro-methoxy]-hydroxy-phosphoryl] phosphono hydrogen phosphate
CN(C)C1=CC=C(C=C1)C2=CC(=NC(=C2)C3=CC=CC=C3OC)C4=CC=CC=C4OC 	4-[2,6-bis(2-methoxyphenyl)-4-pyridyl]-N,N-dimethyl-aniline
C1CN(CC2=CC=CC=C21)C(=O)CC3(CC(=O)N(C3=O)CCC4=CC=CC=N4)C5=CC=CC=C5 	(3R)-3-[2-(3,4-dihydro-1H-isoquinolin-2-yl)-2-oxo-ethyl]-3-phenyl-1-[2-(2-pyridyl)ethyl]pyrrolidine-2,5-dione
CC(C(=O)N(C)C)NCC1=CC=C(C=C1)[Si](C)(C)C 	N,N-dimethyl-2-[(4-trimethylsilylphenyl)methylamino]propanamide
```
* To train from scratch on your custom data, run the following command in the cloned repository. Weight checkpoints will be saved after each training epoch in the ```checkpoints``` folder.
```
python3 train.py --data_path=data/YOUR_TRAINING_DATA.tsv
```
* To finetune the embeddings of a pretrained Transformer on your custom data, provide a path to a pretrained weight file to load before training.
```
python3 train.py --data_path=data/YOUR_TRAINING_DATA.tsv --checkpoint_path=checkpoints/pretrained.ckpt
```
* To train or finetune without GPU resources, provide the ``--cpu``` flag.
```
python3 train.py --data_path=data/YOUR_TRAINING_DATA.tsv --cpu --num_epochs=5.
```

A full list of arguments which control network hyperparameters is provided below, as well as in ```train.py```. The pretrained weights provided were trained using a batch size of 96, split across 4 GPUs, training for 2 epochs on a dataset of 83,000,000+ molecules. The 83m file can be downloaded by running the ```download_large.sh``` script.

```
usage: train.py [-h] [--data_path DATA_PATH]
                [--checkpoint_path CHECKPOINT_PATH] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--max_length MAX_LENGTH]
                [--embedding_size EMBEDDING_SIZE] [--num_layers NUM_LAYERS]
                [--num_epochs NUM_EPOCHS] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to a csv containing pairs of strings for
                        training.
  --checkpoint_path CHECKPOINT_PATH
                        Path to a binary file containing pretrained model
                        weights. If not supplied, a random initialization will
                        be used.
  --batch_size BATCH_SIZE
                        How many samples to average in each training step. If
                        more than one GPU is available, samples will be split
                        across devices.
  --learning_rate LEARNING_RATE
                        Weight updates calculated during gradient descent will
                        be multiplied by this factor before they are added to
                        the weights.
  --max_length MAX_LENGTH
                        Strings in the data longer than this length will be
                        truncated.
  --embedding_size EMBEDDING_SIZE
                        Each SMILES string character will be embedded to a
                        vector with this many elements.
  --num_layers NUM_LAYERS
                        The Encoder and Decoder modules of the Transformer
                        network will each have this many sequential layers.
  --num_epochs NUM_EPOCHS
                        In each epoch, every training sample will be used
                        once.
  --cpu                 Set this flag to run only on the CPU (no cuda needed).

```

## Citing

If you find this work helpful in your research, please cite our publication ["Predicting Binding from Screening Assays with Transformer Network Embeddings"](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01212) 😊.

Preprint: [https://doi.org/10.26434/chemrxiv.11625885](https://doi.org/10.26434/chemrxiv.11625885)

```
@article{morris2020transformer,
author={Morris, Paul
and St. Clair, Rachel
and Hahn, William Edward
and Barenholtz, Elan},
title={Predicting Binding from Screening Assays with Transformer Network Embeddings},
journal={Journal of Chemical Information and Modeling},
year={2020},
month={Jun},
day={22},
publisher={American Chemical Society},
issn={1549-9596},
doi={10.1021/acs.jcim.9b01212},
url={https://doi.org/10.1021/acs.jcim.9b01212}
}
```
