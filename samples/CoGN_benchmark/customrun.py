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
import sys
import pickle
from pathlib import Path

from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
preproc = KNNAsymmetricUnitCell(k=24)
# #matproj base
# trainfile = "mb_mpgammatrain.csv"
# valfile = "mb_mpgammaval.csv"
# testfile = "mb_zbset.csv"
# outfile = "output.csv"
#matrpoj + zb train
#trainfile = "mb_trainset_zbo.csv"
#valfile = "mb_zbset.csv"
#testfile = "mb_zbset.csv"
#outfile = "zbo.csv"
trainfile = sys.argv[1]
valfile = sys.argv[2]
testfile = sys.argv[3]
outroot = sys.argv[4]
try:
    cifdir = sys.argv[5]
except:
    cifdir = "matproj"
try:
    cifdir = sys.argv[6]
except:
    datdir = "../database/"

cachefold = "dataset_cache/"

try:
    with open(cachefold + Path(testfile).stem + ".pkl", "rb") as infile:
        test_dataset = pickle.load(infile)
except (IOError, EOFError):
    test_dataset = CrystalDataset(
        dataset_name="test",
        data_directory=datdir, 
        file_directory=cifdir,
        file_name=testfile
    )
    test_dataset.prepare_data(file_column_name="file", overwrite=False)
    test_dataset.read_in_memory(label_column_name="labels", additional_callbacks={"file": lambda st, ds: ds["file"]})

    test_dataset.set_representation(preproc)
    with open(cachefold + Path(testfile).stem + ".pkl", "wb") as outfile:
        pickle.dump(test_dataset, outfile)

try:
    with open(cachefold + Path(trainfile).stem + ".pkl", "rb") as infile:
        train_dataset = pickle.load(infile)
except (IOError, EOFError):
    train_dataset = CrystalDataset(
        dataset_name="train",
        data_directory=datdir, 
        file_directory=cifdir,
        file_name=trainfile
    )
    train_dataset.prepare_data(file_column_name="file", overwrite=False)
    train_dataset.read_in_memory(label_column_name="labels", additional_callbacks={"file": lambda st, ds: ds["file"]})

    train_dataset.set_representation(preproc)
    with open(cachefold + Path(trainfile).stem + ".pkl", "wb") as outfile:
        pickle.dump(train_dataset, outfile)

try:
    with open(cachefold + Path(valfile).stem + ".pkl", "rb") as infile:
        val_dataset = pickle.load(infile)
except (IOError, EOFError):
    val_dataset = CrystalDataset(
        dataset_name="val",
        data_directory=datdir, 
        file_directory=cifdir,
        file_name=valfile
    )
    val_dataset.prepare_data(file_column_name="file", overwrite=False)
    val_dataset.read_in_memory(label_column_name="labels", additional_callbacks={"file": lambda st, ds: ds["file"]})

    val_dataset.set_representation(preproc)
    with open(cachefold + Path(valfile).stem + ".pkl", "wb") as outfile:
        pickle.dump(val_dataset, outfile)
# We can make a train-test split.
#train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size = 0.2)
#dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]



# Get Labels.
# Make sure the have a label dimension
y_train = np.expand_dims(train_dataset.get("graph_labels"), axis=-1)
y_val = np.expand_dims(val_dataset.get("graph_labels"), axis=-1)
y_test = np.expand_dims(test_dataset.get("graph_labels"), axis=-1)
test_names = np.expand_dims(test_dataset.get("file"), axis=-1)
val_names = np.expand_dims(val_dataset.get("file"), axis=-1)

print("Label shape", y_train.shape, y_test.shape)

# Standardize Labels
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
y_val = scaler.transform(y_val)

# X direct as tensor.
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
x_train = train_dataset.tensor(tensors_for_keras_input)
x_val = val_dataset.tensor(tensors_for_keras_input)
x_test = test_dataset.tensor(tensors_for_keras_input)

print("Feautres shape", {key: value.shape for key, value in x_train.items()})


# Get the model
model = make_model(
    name = "coGN",
    inputs = tensors_for_keras_input
    # All defaults else
 )

# Compile the mode with loss and metrics.
model.compile(
    loss="mean_absolute_error",
    optimizer=Adam(
        learning_rate=KerasPolynomialDecaySchedule(
            dataset_size=80, batch_size=64, epochs=800,
            lr_start=0.0005, lr_stop=1.0e-05
        )
    ),
    metrics=["mean_absolute_error"], # Note targets are standard scaled.
)

#model_best = ModelCheckpoint(outroot[:-4] + ".model", save_best_only=True, monitor="mean_absolute_error", mode='min')
earlystop = EarlyStopping(monitor="val_loss", mode='min', patience = 50, restore_best_weights = True)
# Fit model.
model.fit(
    x_train,
    y_train,
    callbacks=[
        earlystop
        # We can use schedule instead of scheduler ...
        # LinearLearningRateScheduler(epo_min=10, epo=1000, learning_rate_start=5e-04, learning_rate_stop=1e-05)
    ],
    validation_data=(x_val, y_val),
    validation_freq=1,
    shuffle=True,
    batch_size=64,
    epochs=800,
    verbose=2,
)


#model.load_weights(outroot[:-4] + ".model") 
# Model prediction
predict_val = scaler.inverse_transform(model.predict(x_val))
y_val = scaler.inverse_transform(y_val)
print("Error:", np.mean(np.abs(predict_val - y_val)))
outs = np.hstack([val_names, predict_val, y_val])
outdat = pd.DataFrame(outs, columns = ["file","predicted","target"])
outdat.to_csv(outroot[:-4] + "_val.csv", index = False)

predict_test = scaler.inverse_transform(model.predict(x_test))
y_test = scaler.inverse_transform(y_test)
print("Error:", np.mean(np.abs(predict_test - y_test)))
outs = np.hstack([test_names, predict_test, y_test])
outdat = pd.DataFrame(outs, columns = ["file","predicted","target"])
outdat.to_csv(outroot, index = False)

with open(outroot[:-4] + "_mdl.pkl", "wb") as outfile:
    pickle.dump(model, outfile)
model.save(outroot[:-4] + "_mdl.h5")