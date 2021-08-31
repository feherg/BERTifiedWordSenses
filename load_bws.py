import pandas as pd
import numpy as np
import glob
from itertools import chain
from sklearn.preprocessing import normalize
import ast

def load_data(path):
    # NOTE: expects the filepath or path to bws files, i.e., the output of bertify_ufsac.py
    if ".csv.xz" in path:
        # we have one file
        return pd.read_csv(path, index_col=None, header=0)
    else:
        # several files
        all_files = glob.glob(path + "/*.csv.xz")

        l = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            l.append(df)

        data = pd.concat(l, axis=0, ignore_index=True)
        return data


# %%
def clean_data(data):
    # merge embedding columns into np array
    data["avg"] = np.vsplit(data[data.columns[7:-1]].values, len(data)) 
    data["avg"] = data["avg"].apply(lambda x: x.flatten()) # get rid of unnecessary first dim; shape is now (768,)
    data.drop(columns=[str(x) for x in range(768)], inplace=True) # remove 
    data.drop(columns=["remove"], inplace=True)
    data["wn_30_sense"]=data["wn_30_sense"].apply(lambda x: ast.literal_eval(x)) # read list
    data["senseID"]=data["senseID"].apply(lambda x: ast.literal_eval(x)) # read list

    # unique senses in the dataset
    unique = set(list(chain.from_iterable(data["wn_30_sense"].values.tolist())))


    return data, len(unique)


# %%
def get_normalized_vectors(data):
    X = normalize(np.vstack(data["avg"].values), axis=1)
    return X