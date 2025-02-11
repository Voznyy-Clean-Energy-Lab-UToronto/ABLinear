
import pandas as pd
import random
import numpy as np
import sys


inp = pd.read_csv("zb_gamma cp2k.csv")
inp = inp[~inp.atom1.isin(["Cu","Zn","Ag","Cd","Au","Hg"])]
inp = inp[~inp.atom1.isin(["B","Al","Ga","In","Tl"])]

a1 = inp.atom1.unique()
a2 = inp.atom2.unique()
labelset = inp.label.unique()

def tbwrite(filename, train, val):
    train.to_csv(filename+"ban.csv", index = 0)
    val.to_csv(filename+"test.csv", header = 0, index = 0)

for i in a1:
    tset = inp[inp.atom1 != i]
    vset = inp[inp.atom1 == i]
    tbwrite("tb2/tb_{}_".format(i), tset, vset)

for i in a2:
    tset = inp[inp.atom2 != i]
    vset = inp[inp.atom2 == i]
    tbwrite("tb2/tb_{}_".format(i), tset, vset)

for i in labelset:
    tset = inp[inp.label != i]
    vset = inp[inp.label == i]
    tbwrite("tb2/tb_{}_".format(i), tset, vset)

print('"', end="")
print(*a1, *a2, *labelset, sep = "\" \"", end = '"\n')