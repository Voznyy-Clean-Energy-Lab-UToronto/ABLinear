
import pandas as pd
import random
import numpy as np
import sys

if len(sys.argv)>2:
    split = int(sys.argv[2])
else:
    split = 80
frac = split / 100

for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zb_gamma cp2k.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbg_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbg_val_{}.csv".format(split,seed), header = 0, index = 0)

for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zb_homo-p offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbh_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbh_val_{}.csv".format(split,seed), header = 0, index = 0)
    
for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zb_lumo-s offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbl_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbl_val_{}.csv".format(split,seed), header = 0, index = 0)
    
    
for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zba_gamma cp2k.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbag_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbag_val_{}.csv".format(split,seed), header = 0, index = 0)

for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zba_homo-p offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbah_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbah_val_{}.csv".format(split,seed), header = 0, index = 0)
    
for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zba_lumo-s offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbal_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbal_val_{}.csv".format(split,seed), header = 0, index = 0)
    

for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zbai_gamma cp2k.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbaig_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbaig_val_{}.csv".format(split,seed), header = 0, index = 0)

for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zbai_homo-p offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbaih_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbaih_val_{}.csv".format(split,seed), header = 0, index = 0)
    
for seed in range(1,int(sys.argv[1])+1):
    inp = pd.read_csv("zbai_lumo-s offset.csv")
    inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
    inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

    train, val = np.split(inp.sample(frac=1, random_state = seed), [int(frac*len(inp))])

    train.to_csv("{}zbail_train_{}.csv".format(split, seed), header = 0, index = 0)
    val.to_csv("{}zbail_val_{}.csv".format(split,seed), header = 0, index = 0)