
import pandas as pd
import random
import numpy as np
import sys

if len(sys.argv)>2:
    split = int(sys.argv[2])
else:
    split = 5
seed = int(sys.argv[1])

droplist = ["Mg_O", "Na_F", "Rb_Br", "Li_I"]

def splitter(ifilename, refname, flag, drop):
    inp = pd.read_csv(ifilename)
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]
    
    test = inp[inp.label.isin(drop)]
    inp = inp[~inp.label.isin(drop)]

    fracs = np.array_split(inp.sample(frac=1, random_state = seed), split)
    #print(fracs)
    
    refset = pd.read_csv(refname)
    uref = refset[~refset.label.isin(["rs_" + i for i in drop])]

    for i in range(len(fracs)):
        val = fracs[i]
        train = pd.concat([fracs[j] for j in range(len(fracs)) if j != i])
        train_uref = pd.concat([train, uref])
        train_ref = pd.concat([train, refset])

        train_ref.to_csv("{}_fold_{}_{}r_train_{}.csv".format(split, drop[0], flag, i), header = 0, index = 0)
        train_uref.to_csv("{}_fold_{}_{}u_train_{}.csv".format(split, drop[0], flag, i), header = 0, index = 0)
        val.to_csv("{}_fold_{}_{}_val_{}.csv".format(split, drop[0], flag,i), header = 0, index = 0)
    test.to_csv("{}_fold_{}_{}_test.csv".format(split, drop[0], flag), header = 0, index = 0)

for i in droplist:
    splitter("zb_gap.csv", "rso_gap.csv", "rsallback", [i])