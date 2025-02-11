
import pandas as pd
import random
import numpy as np
import sys

if len(sys.argv)>2:
    split = int(sys.argv[2])
else:
    split = 5
seed = int(sys.argv[1])

def splitter(ifilename, flag):
    inp = pd.read_csv(ifilename)
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    fracs = np.split(inp.sample(frac=1, random_state = seed), split)
    #print(fracs)

    for i in range(len(fracs)):
        val = fracs[i]
        train = pd.concat([fracs[j] for j in range(len(fracs)) if j != i])

        train.to_csv("{}_fold_zb_{}_train_{}.csv".format(split, flag, i), header = 0, index = 0)
        val.to_csv("{}_fold_zb_{}_val_{}.csv".format(split, flag,i), header = 0, index = 0)

splitter("zb_ovr.csv", "ovr") #atomic s-s splitting
splitter("zb_ovr_splt.csv", "ovrs") #atomic s-s splitting
splitter("zb_ovr_delt.csv", "ovrd") #atomic s-s splitting
splitter("zb_ovr_only.csv", "ovro") #atomic s-s splitting