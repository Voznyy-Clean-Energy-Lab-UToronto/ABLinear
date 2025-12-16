
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
#import h5py
import json

#from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
#import tensorflow.keras as ks

from matbench.bench import MatbenchBenchmark
from graphlist import GraphList, HDFGraphList
from kgcnn.literature.coGN import make_model, model_default, model_default_nested
from kgcnn.crystal.preprocessor import KNNUnitCell, KNNAsymmetricUnitCell, CrystalPreprocessor, VoronoiAsymmetricUnitCell
from kgcnn.graph.methods import get_angle_indices
from preprocessing import MatbenchDataset

import pickle

cifprefix = "cifs"


mb = MatbenchBenchmark(subset = ["matbench_mp_e_form"])
mb.load()
for task in mb.tasks:
    for split in range(5):
        train_inputs, train_outputs = task.get_train_and_val_data(split)
        test_inputs, test_outputs = task.get_test_data(split, include_target=True)

        with open("train_inputs_ef.pkl", "wb") as outfile:
            pickle.dump(train_inputs, outfile)
        with open("test_inputs_ef.pkl", "wb") as outfile:
            pickle.dump(test_inputs, outfile)
        train_outputs.to_csv("train_ef_{}.csv".format(split))
        test_outputs.to_csv("test_ef_{}.csv".format(split))