import pickle
from tensorflow.keras.models import Model, load_model
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import tensorflow as tf
import numpy as np
from kgcnn.data.crystal import CrystalDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from kgcnn.literature.coGN import make_model
from kgcnn.training.schedule import KerasPolynomialDecaySchedule
from kgcnn.training.scheduler import LinearLearningRateScheduler
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from pathlib import Path
from adjustText import adjust_text
from sklearn.decomposition import PCA

cachefold = "dataset_cache/"
testfile = "mb_cscl_rs_zb"
modelname = "cscl_rs_zb_mdl"

#for making UMAPS from coGN, requires a pickle of processed coGN cifs

with open(cachefold + testfile + ".pkl", "rb") as infile:
    test_dataset = pickle.load(infile)

test_names = np.expand_dims(test_dataset.get("file"), axis=-1)
y_test = np.expand_dims(test_dataset.get("graph_labels"), axis=-1)
tensors_for_keras_input = {
    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
    "cell_translation": None,
    "affine_matrix": None,
    "voronoi_ridge_area": None,
    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
    "frac_coords": None,
    "coords": None,
    "multiplicity": {"shape": (None,), "name": "multiplicity", "dtype": "int32", "ragged": True},
    "lattice_matrix": None,
    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int32", "ragged": True},
    "line_graph_edge_indices": None,
}
x_test = test_dataset.tensor(tensors_for_keras_input)

hards = pd.read_csv("cscl_rs_zb.csv")
hards["group"] = hards["file"].str[0]
hards["col"] = hards["group"].map({"c":"r", "r":"b", "z":"g"})

model = make_model(
    name = "coGN",
    inputs = tensors_for_keras_input
    # All defaults else
 )
model.load_weights(modelname + ".h5") 


embedding_layer = model.get_layer("graph_network_multiplicity_readout")
embedding_model = Model(
    inputs=model.input,
    outputs=embedding_layer.output[0]
)

X_model = embedding_model.predict(x_test)
predict_test = model.predict(x_test)
errors = np.abs(predict_test - y_test)   


X_graph = tf.reduce_mean(X_model, axis=1).numpy()


reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

X_umap = reducer.fit_transform(X_graph)

plt.figure()
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=errors, s = 6, linewidths = 0, alpha = 0.5)  # optional labels
plt.colorbar(scatter, label="Prediction Error")
plt.title("UMAP of model embeddings colored by prediction error")
plt.savefig("umap_crz.png")

plt.figure()
fig, ax = plt.subplots()
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=hards["col"], s = 6, linewidths = 0, alpha = 1)  # optional labels
legendset = [Line2D([0], [0], marker='o', c="r", label='CsCl', markersize=6),
    Line2D([0], [0], marker='o', c="g", label='Rocksalt', markersize=6),
    Line2D([0], [0], marker='o', c="b", label='Zincblende', markersize=6)]
ax.legend(handles = legendset, loc = "center right")
plt.title("UMAP of model embeddings colored by structure")
plt.savefig("umap_crz_type.png")


df = pd.DataFrame(
    {"file": test_names.flatten(),
    "umap_x": X_umap[:, 0],
    "umap_y": X_umap[:, 1],
    "error": errors.flatten(),
    "group": hards["group"].values,
    "color": hards["col"].values,
    "gap": y_test.T[0]}
    )

# Save CSV
df.to_csv("umap_crz.csv", index=False)

hards["file"] = hards["file"].str.replace(r'\.[^.]+$', '', regex=True)
split_cols = hards["file"].str.split('_', expand=True)
hards["cation"] = split_cols[1]
hards["anion"] = split_cols[2]
color_labels = hards['cation'].unique()
shape_labels = hards['anion'].unique()
colors = plt.cm.tab20.colors
markers = ['o','s','^','D','v','<','>','P','X','*','H','+']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(color_labels)}
marker_map = {label: markers[i % len(markers)] for i, label in enumerate(shape_labels)}
hards["umapx"] = X_umap[:, 0]
hards["umapy"] = X_umap[:, 1]
plt.figure()
fig, ax = plt.subplots()
for (c, s), group in hards.groupby(['cation','anion']):
    ax.scatter(group['umapx'], group['umapy'],
               color=color_map[c],
               marker=marker_map[s],
               s=10)
color_legend = [Line2D([0],[0], marker='o', color='w', label=label,
                       markerfacecolor=color_map[label], markersize=10)
                for label in color_labels]
shape_legend = [Line2D([0],[0], marker=marker_map[label], color='k', label=label,
                        linestyle='None', markersize=10)
                for label in shape_labels]
legend1 = ax.legend(handles=color_legend, title='Cation', loc='upper left', bbox_to_anchor=(1,1))
legend2 = ax.legend(handles=shape_legend, title='Anion', loc='upper left', bbox_to_anchor=(2,1))
plt.title("UMAP of model embeddings by atoms")
plt.tight_layout()
plt.savefig("umap_crz_atm.png")

reducer = PCA(.95)

X_umap = reducer.fit_transform(X_graph)
expl_var = reducer.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

print("Explained variance ratio:", expl_var)
print("Cumulative variance ratio:", cum_var)

plt.figure()
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=errors, s = 6, linewidths = 0, alpha = 0.5)  # optional labels
plt.colorbar(scatter, label="Prediction Error")
plt.title("PCA of model embeddings colored by prediction error")
plt.savefig("pca_crz.png")

plt.figure()
fig, ax = plt.subplots()
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=hards["col"], s = 6, linewidths = 0, alpha = 1)  # optional labels
legendset = [Line2D([0], [0], marker='o', c="r", label='CsCl', markersize=6),
    Line2D([0], [0], marker='o', c="g", label='Rocksalt', markersize=6),
    Line2D([0], [0], marker='o', c="b", label='Zincblende', markersize=6)]
ax.legend(handles = legendset, loc = "center right")
plt.title("PCA of model embeddings colored by structure")
plt.savefig("pca_crz_type.png")


df = pd.DataFrame(
    {"file": test_names.flatten(),
    "umap_x": X_umap[:, 0],
    "umap_y": X_umap[:, 1],
    "error": errors.flatten(),
    "group": hards["group"].values,
    "color": hards["col"].values,
    "gap": y_test.T[0]}
    )

# Save CSV
df.to_csv("pca_crz.csv", index=False)