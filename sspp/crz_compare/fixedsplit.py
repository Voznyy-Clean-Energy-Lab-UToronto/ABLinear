
import pandas as pd
import numpy as np

testlist = ["Mg_O", "Na_F"]

vallist = ["Na_Br", "Au_F", "K_I", "Be_O", "Ca_S", "Ag_Br", "Ca_O", "Ga_P"]

COLUMN_ORDER = ["file", "labels", "cation", "anion", "n"] 
COLUMN_ORDER_H = ["file", "hartree", "cation", "anion", "n"] 

RNG_SEED = 123
rng = np.random.default_rng(RNG_SEED)

STRUCTURE_SETS = {
    "zb": ["zb"],
    "zb_rs": ["zb", "rs"],
    "zb_rs_cscl": ["zb", "rs", "cscl"],
}


df = pd.read_csv("mb_cscl_rs_zb.csv")

parsed = df["file"].str.replace(".cif", "", regex=False).str.split("_", expand=True)

df["structure"] = parsed[0]
df["compound"]  = parsed[1] + "_" + parsed[2]
df["hartree"] = df["labels"] / 27.2114079527

df_zb   = df[df["structure"] == "zb"]
df_rs   = df[df["structure"] == "rs"]
df_cscl = df[df["structure"] == "cscl"]

excluded = set(vallist + testlist)

zb_val  = df_zb[df_zb["compound"].isin(vallist)]
zb_test = df_zb[df_zb["compound"].isin(testlist)]

zb_train_inclusive = df_zb[
    ~df_zb["compound"].isin(excluded)
]
zb_train_exclusive = df_zb[
    ~df_zb["compound"].isin(excluded)
]

train_zb_inclusive = zb_train_inclusive
train_zb_exclusive = zb_train_exclusive

zb_val_compounds = set(vallist + testlist)

def sample_n_compounds(df, n, forbidden, rng):
    candidates = (
        df[~df["compound"].isin(forbidden)]
        ["compound"]
        .unique()
    )

    if len(candidates) < n:
        raise ValueError(
            f"Not enough candidates: need {n}, have {len(candidates)}"
        )

    return set(rng.choice(candidates, size=n, replace=False))

rs_val_compounds = sample_n_compounds(
    df_rs,
    n=8,
    forbidden=zb_val_compounds,
    rng=rng
)

# cscl: exclude zb + rs validation
cscl_val_compounds = sample_n_compounds(
    df_cscl,
    n=8,
    forbidden=zb_val_compounds | rs_val_compounds,
    rng=rng
)

rs_val   = df_rs[df_rs["compound"].isin(rs_val_compounds)]
cscl_val = df_cscl[df_cscl["compound"].isin(cscl_val_compounds)]

rs_train = df_rs[~df_rs["compound"].isin(rs_val_compounds)]
cscl_train = df_cscl[~df_cscl["compound"].isin(cscl_val_compounds)]


assert len(rs_val_compounds) == 8
assert len(cscl_val_compounds) == 8

assert not (zb_val_compounds & rs_val_compounds)
assert not (zb_val_compounds & cscl_val_compounds)
assert not (rs_val_compounds & cscl_val_compounds)

train_zb_rs_inclusive = pd.concat([
    zb_train_inclusive,
    rs_train
])

train_zb_rs_exclusive = pd.concat([
    zb_train_exclusive,
    rs_train[~rs_train["compound"].isin(testlist)]
])

train_zb_rs_cscl_inclusive = pd.concat([
    zb_train_inclusive,
    rs_train,
    cscl_train
])

train_zb_rs_cscl_exclusive = pd.concat([
    zb_train_exclusive,
    rs_train[~rs_train["compound"].isin(testlist)],
    cscl_train[~cscl_train["compound"].isin(testlist)]
])

zb_rs_val = pd.concat([zb_val, rs_val])
zb_rs_cscl_val = pd.concat([zb_val, rs_val, cscl_val])

def finalize(d, ord = COLUMN_ORDER):
    return (
        d
        .drop(columns=["structure", "compound"], errors="ignore")
        .reindex(columns=ord)
    )
    
splits = {
    "zb": {
        "inclusive": {
            "train": finalize(train_zb_inclusive),
            "val":   finalize(zb_val),
            "test":  finalize(zb_test),
        },
        "exclusive": {
            "train": finalize(train_zb_exclusive),
            "val":   finalize(zb_val),
            "test":  finalize(zb_test),
        },
    },
    "zb_rs": {
        "inclusive": {
            "train": finalize(train_zb_rs_inclusive),
            "val":   finalize(zb_rs_val),
            "test":  finalize(zb_test),
        },
        "exclusive": {
            "train": finalize(train_zb_rs_exclusive),
            "val":   finalize(zb_rs_val),
            "test":  finalize(zb_test),
        },
    },
    "zb_rs_cscl": {
        "inclusive": {
            "train": finalize(train_zb_rs_cscl_inclusive),
            "val":   finalize(zb_rs_cscl_val),
            "test":  finalize(zb_test),
        },
        "exclusive": {
            "train": finalize(train_zb_rs_cscl_exclusive),
            "val":   finalize(zb_rs_cscl_val),
            "test":  finalize(zb_test),
        },
    },
}

for structure, d in splits.items():
    for mode in ["inclusive", "exclusive"]:
        for split in ["train", "val", "test"]:
            d[mode][split].to_csv(
                f"{structure}_{mode}_{split}.csv",
                index=False, header = False
            )
            
splits = {
    "zb": {
        "inclusive": {
            "train": finalize(train_zb_inclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
        "exclusive": {
            "train": finalize(train_zb_exclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
    },
    "zb_rs": {
        "inclusive": {
            "train": finalize(train_zb_rs_inclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
        "exclusive": {
            "train": finalize(train_zb_rs_exclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
    },
    "zb_rs_cscl": {
        "inclusive": {
            "train": finalize(train_zb_rs_cscl_inclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
        "exclusive": {
            "train": finalize(train_zb_rs_cscl_exclusive, COLUMN_ORDER_H),
            "val":   finalize(zb_val, COLUMN_ORDER_H),
            "test":  finalize(zb_test, COLUMN_ORDER_H),
        },
    },
}

for structure, d in splits.items():
    for mode in ["inclusive", "exclusive"]:
        for split in ["train", "val", "test"]:
            d[mode][split].to_csv(
                f"{structure}_{mode}_{split}2.csv",
                index=False, header = False
            )