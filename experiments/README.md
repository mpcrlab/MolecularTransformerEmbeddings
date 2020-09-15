# Experiment Code

The Python Jupyter notebooks in this repository contain code used in the paper experiments.

### Setup 
* ```download_binding_datasets.sh```: Grabs TSV files for two assays that were too large to include in the repository.
* ```embedding_binaries/download_instructions.txt```: Instructions to download large, preprocessed hdf5 embedding binaries used to train classifiers. The files can be downloaded from the following google drive link and manually placed in the ```embedding_binaries``` folder. ```hiv1_protease_2.hdf5``` is the smallest (~3Gb) and easiest dataset. Optionally, the larger two files (both ~200Gb) can be built using the ```build_embedding_binaries.ipynb``` notebook.
https://drive.google.com/drive/folders/1mJZPlFaStmsuwnpGVlpwJ7XL9Di_6j5u?usp=sharing

### Notebooks
* ```embedding_classifier_final.ipynb```: Classifies molecular embeddings in an hdf5 dataset and calculates AUC.
* ```build_embedding_binaries.ipynb```: Reads a TSV file in the ```binding_datasets``` folder and generates a preprocessed hdf5 binary in the ```embedding_binaries``` folder. Requires the pretrained transformer checkpoint be downloaded (this can be done using the ```download.sh``` script in the root of the repository)
* ```dataset_similarity.ipynb```: Calculates Tanimoto similarity between train and test cross-validation splits for a dataset to judge its difficulty in classification problems.
* ```analyze_reactions.ipynb```: Compares pairs of molecule embeddings with their predicted binding confidence to a target.
* ```analyze_embeddings.ipynb```: Embeds SMILES strings and performs dimensionality reduction for visualization.
