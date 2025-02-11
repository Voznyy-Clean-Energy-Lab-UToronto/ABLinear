
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

# splitter("zba_ss.csv", "ss") #atomic s-s splitting
# splitter("zba_dss.csv", "dss") #atomic s-s delta
# splitter("zba_pp.csv", "pp") #atomic p-p splitting
# splitter("zba_dpp.csv", "dpp") #atomic p-p delta

# splitter("zb_iss.csv", "iss") #ionic s-s splitting
# splitter("zb_diss.csv", "diss") #ionic s-s delta
# splitter("zb_ipp.csv", "ipp") #ionic p-p splitting
# splitter("zb_dipp.csv", "dipp") #ionic p-p delta


# splitter("zbai_ss.csv", "ai_ss") #atomic s-s splitting
# splitter("zbai_dss.csv", "ai_dss") #atomic s-s delta
# splitter("zbai_pp.csv", "ai_pp") #atomic p-p splitting
# splitter("zbai_dpp.csv", "ai_dpp") #atomic p-p delta

# splitter("zbai_iss.csv", "ai_iss") #ionic s-s splitting
# splitter("zbai_diss.csv", "ai_diss") #ionic s-s delta
# splitter("zbai_ipp.csv", "ai_ipp") #ionic p-p splitting
# splitter("zbai_dipp.csv", "ai_dipp") #ionic p-p delta

splitter("zb_dippr.csv", "dippr") #ionic p-p delta
splitter("zb_dissr.csv", "dissr") #ionic s-s delta