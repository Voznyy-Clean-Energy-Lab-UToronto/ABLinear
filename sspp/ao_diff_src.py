
import pandas as pd
import random
import numpy as np
import sys

inp = pd.read_csv("zb_src.csv")
inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]
# #ion orbs
atomics = pd.read_csv("bigcell_ion_He_ref_tn2.csv")

abs_val = inp.merge(atomics, how = "left", left_on = "atom1", right_on = "atom")
abs_val = abs_val.merge(atomics, how = "left", left_on = "atom2", right_on = "atom")
abs_val = abs_val.drop(["atom_x","atom_y"], axis = 1)

cols_i = abs_val.columns.tolist()[13:]
for i in cols_i:
    abs_val[i] -= abs_val.s1_x
#abs_val = abs_val.drop(["s1_x","d1_x","d2_x","d3_x","d4_x","d5_x"], axis = 1)
abs_val = abs_val.drop(["s1_x","d1_x","d2_x"], axis = 1)
#print(abs_val)

cols = abs_val.columns.tolist()

targets = cols[4:12]
label = cols[0]
atoms = cols[1:3]
struct = cols[3]
orbs = cols[12:]

for i in targets:
    filtset = [label] + [i] + atoms + orbs + [struct]
    abs_val[filtset].to_csv("zb_{}.csv".format(i), index = 0)

#atomic orbs
atomics = pd.read_csv("bigcell_He_ref_tn.csv")

abs_val_atomic = inp.merge(atomics, how = "left", left_on = "atom1", right_on = "atom")
abs_val_atomic = abs_val_atomic.merge(atomics, how = "left", left_on = "atom2", right_on = "atom")
abs_val_atomic = abs_val_atomic.drop(["atom_x","atom_y"], axis = 1)

cols_a = abs_val_atomic.columns.tolist()[13:]

for i in cols_a:
    abs_val_atomic[i] -= abs_val_atomic.s1_x
abs_val_atomic = abs_val_atomic.drop(["s1_x","d1_x","d2_x","d3_x","d4_x","d5_x"], axis = 1)

cols_a = abs_val_atomic.columns.tolist()

targets = cols_a[4:12]
label = cols_a[0]
atoms = cols_a[1:3]
struct = cols_a[3]
orbs = cols_a[12:]

for i in targets:
    filtset = [label] + [i] + atoms + orbs + [struct]
    abs_val_atomic[filtset].to_csv("zba_{}.csv".format(i), index = 0)
    
    
abs_val_ai = abs_val.merge(abs_val_atomic[[label]+orbs], how = "left", on = "label")
cols_ai = abs_val_ai.columns.tolist()

targets = cols_ai[4:12]
label = cols_ai[0]
atoms = cols_ai[1:3]
struct = cols_ai[3]
orbs = cols_ai[12:]
for i in targets:
    filtset = [label] + [i] + atoms + orbs + [struct]
    abs_val_ai[filtset].to_csv("zbai_{}.csv".format(i), index = 0)
