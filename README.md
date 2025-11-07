# README

ABLinear is a simplified set of models aimed to deconstruct graph neural networks for crystalline materials. 
The models operate by selectively invoking the components of a GNN on binary materials. 
The standard input is to take two atoms, cation and anion, and embed their features before passing them to the model.

## Basic operation

ABLinear_nn.py trains and ABLinear_pred.py predicts.

python ABLinear_nn.py RUN_DIRECTORY [--id-prop-t TRAINING_CSV] [--id-prop-v VALIDATION_CSV] [--out OUTPUT_FOLDER] [-m Model_Type] [-e EPOCHS] [-b BATCH_SIZE] [--lr LEARNING_RATE] [--wd WEIGHT_DECAY] [--width MODEL_WIDTH] [--ari ATOM_FEATURES_JSON]

Model types are as follows:

| Model number | Contents |
| --- | --- |
| 0 | Linear layers only |
| 1 | Convolution + linear |
| 2 | Convolution + pooling + linear |
| 3 | Bilinear convolution + pooling + linear |
| 4 | Graph convolution + linear |
| 5 | Graph convolution + pooling + linear |
| 6 | Graph convolution (neighbours weighted) + pooling + linear |
| 7 | Pooling + linear |
| 8 | Linear layers (no activation) |

python ABLinear_pred.py RUN_DIRECTORY [--id-prop-p PREDICTING_CSV] [--out OUTPUT_FOLDER] [-m Model_Type] [-b BATCH_SIZE] [--width MODEL_WIDTH] [--ari ATOM_FEATURES_JSON]

## CSV formats

Headerless CSVs are used for training/predicting. 
Layout is label, target, atom1, atom2, additional features ... , class. 
The class is treated as number of neighbours for model type 6.

## Navigation

ABLinear/SSPP contains the sample training files for the model as governed by ABLinear/samples which contain the batch files for replication of work.
Header names are found in the source files and python utilities.
