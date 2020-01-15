import os
import argparse
import time
import torch
import numpy as np

from transformer import Transformer, create_masks
from load_data import ALPHABET_SIZE, EXTRA_CHARS


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="data/amino_acids.txt", help="Path to a text file with one SMILES string per line. These strings will be embedded.")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/pretrained.ckpt", help="Path to a binary file containing pretrained model weights.")
parser.add_argument("--max_length", type=int, default=256, help="Strings in the data longer than this length will be truncated.")
parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size used in the pretrained Transformer.")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers used in the Encoder and Decoder of the pretrained Transformer.")

args = parser.parse_args()

print(args)

def encode_char(c):
    return ord(c) - 32

def encode_smiles(string, start_char=EXTRA_CHARS['seq_start']):
    return torch.tensor([ord(start_char)] + [encode_char(c) for c in string], dtype=torch.long)[:args.max_length].unsqueeze(0)


smiles_strings = [line.strip("\n") for line in open(args.data_path, "r")]
print("Loaded {0} SMILES strings from {1}".format(len(smiles_strings), args.data_path))

print("Initializing Transformer...")
model = Transformer(ALPHABET_SIZE, args.embedding_size, args.num_layers)
print("Transformer Initialized.")

print("Loading pretrained weights from", args.checkpoint_path)
checkpoint = torch.load(args.checkpoint_path)
try:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
except AttributeError as e:
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
print("Pretrained weights loaded")

try:
    encoder = model.module.encoder
except AttributeError as e:
    encoder = model.encoder

embeddings = []
with torch.no_grad():
    for smiles in smiles_strings:
        encoded = encode_smiles(smiles)
        mask = create_masks(encoded)
        embedding = encoder(encoded, mask)[0].numpy()
        embeddings.append(embedding)
        print("embedded {0} into {1} matrix.".format(smiles, str(embedding.shape)))
        
print("All SMILES strings embedded. Saving...")
filename = os.path.splitext(os.path.basename(args.data_path))[0]
out_dir = "embeddings/"
out_file = os.path.join(out_dir, filename + ".npz")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_dict = {smiles: matrix for smiles, matrix in zip(smiles_strings, embeddings)}
np.savez(out_file, **out_dict)
print("Saved embeddings to", out_file)

