
import umap
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pymatgen.core import Structure
from matminer.featurizers.composition import ElementProperty
import pickle

datpath = "database/efcifs/"
gap_pred = "All_mb_formation.csv"

#Matminer magpie neighbour analyzer, requires cifs and a csv of labels, targets, and predictions from a given model

df = pd.read_csv(gap_pred)
try:
    with open(gap_pred[:-4] + "magpie.pkl", "rb") as infile:
        features, valid_idx = pickle.load(infile)
except:
    paths = (datpath + df["file"]).tolist()

    magpie = ElementProperty.from_preset("magpie")

    features = []
    valid_idx = []   # keep track of rows that successfully featurize

    for i, cif in enumerate(paths):
        print(i)
        try:
            s = Structure.from_file(cif)
            comp = s.composition
            feats = magpie.featurize(comp)
            features.append(feats)
            valid_idx.append(i)
        except Exception as e:
            print(f"Failed to featurize {cif}: {e}")
    with open(gap_pred[:-4] + "magpie.pkl", "wb") as outfile:
        pickle.dump([features, valid_idx],outfile)
    
        
feature_df = pd.DataFrame(features)
feature_df.index = df.index[valid_idx]

# Merge with metadata
full = df.loc[valid_idx]
full_features = pd.concat([full, feature_df], axis=1).reset_index(drop=True)

# only numeric feature columns (skip metadata)
X = feature_df.values

y_true = np.array(full_features["target"])
k = 5
nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)

distances, indices = nbrs.kneighbors(X)

# Remove self-neighbor (index 0)
neighbor_indices = indices[:, 1:]

# Compute local roughness: std dev of neighbor ground-truth values
local_average = np.array([np.mean(y_true[neighbor_indices[i]])for i in range(len(y_true))])
local_roughness = y_true - local_average
local_std = np.array([np.std(y_true[indices[i]])for i in range(len(y_true))])

dfo = pd.DataFrame(
    {"file": full_features["file"],
    "target": full_features["target"],
    "pred": full_features["predicted"],
    "roughness": local_roughness,
    "local_avg": local_average,
    }
    )

# Save CSV
dfo.to_csv("pca_matminer_ef_magpie.csv", index=False)