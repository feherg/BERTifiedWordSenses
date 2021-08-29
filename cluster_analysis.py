# %%
import pandas as pd
import numpy as np
import glob
from itertools import chain
from collections import Counter
from sklearn.preprocessing import normalize
import hdbscan
from sklearn import metrics
import skfuzzy as fuzz
import os
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

# %%
def initial_centers(tfidf, k, seed):
    return tfidf[np.random.default_rng(seed=seed).choice(tfidf.shape[0], k, replace=False)]

# %%
def sphericalkmeans(tfidf, centers, max_iter=100):
    """REQUIRES the input to be L2 normalized, and does not handle corner cases such as empty clusters!"""
    last_assignment = None
    for iter in range(max_iter):
        assignment = np.asarray((tfidf @ centers.T).argmax(axis=1)).ravel()
        if last_assignment is not None and (assignment == last_assignment).all(): break
        last_assignment = assignment
        centers = np.zeros(centers.shape)
        for i in range(centers.shape[0]):
            c = np.asarray(tfidf[assignment == i,:].sum(axis=0)).ravel()
            centers[i] = c / np.sqrt((c**2).sum())
    return centers, last_assignment, iter


# %%
def cluster_spherical(X, k, seed, gt, outpath):
    c = initial_centers(X, k, seed)
    c, l, _ = sphericalkmeans(X, c, max_iter=100)

    pu, re, f1 = prf1(l, gt)

    # filter out data points with more than one label 
    single_idx = [i for i in range(len(gt)) if len(gt[i]) == 1]
    if len(single_idx) > 0:
        # there are points with only 1 label
        gt = np.asarray(gt, dtype=object).flatten()

        ari = metrics.adjusted_rand_score(gt[single_idx], l[single_idx])
        nmi = metrics.normalized_mutual_info_score(gt[single_idx], l[single_idx])
    else:
        # no points with only one label
        # ari & nmi are not defined
        ari = np.nan
        nmi = np.nan

    with open(outpath+"_skm.csv", "a") as f:
        f.write(";".join([str(k), str(seed), str(ari), str(nmi), str(pu), str(re), str(f1)]) + "\n")

# %%
def cluster_hdbscan(X, gt, outpath):
    X_cosine_dists = metrics.pairwise.pairwise_distances(X, metric="cosine")
    hdbs = hdbscan.HDBSCAN(metric='precomputed')
    hdbs.fit(X_cosine_dists)

    pu, re, f1 = prf1(hdbs.labels_, gt)

    # filter out data points with more than one label 
    single_idx = [i for i in range(len(gt)) if len(gt[i]) == 1]
    if len(single_idx) > 0:
        # there are points with only 1 label
        gt = np.asarray(gt, dtype=object).flatten()

        ari = metrics.adjusted_rand_score(gt[single_idx], hdbs.labels_[single_idx])
        nmi = metrics.normalized_mutual_info_score(gt[single_idx], hdbs.labels_[single_idx])
    else:
        # no points with only one label
        # ari & nmi are not defined
        ari = np.nan
        nmi = np.nan


    with open(outpath+"_est_ks_hdbs.csv", "a") as f:
        f.write(";".join([str(len(np.unique(hdbs.labels_))), str(np.nan), str(ari), str(nmi), str(pu), str(re), str(f1)]) + "\n")
    

def cluster_fuzzy_cmeans(X, k, seed, gt, outpath):
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(X.T, k, 2, error=0.005, maxiter=1000, init=None, seed=seed)
    # NOTE: u.shape == (k, len(X))

    labels = np.argmax(u, axis=0)

    pu, re, f1 = prf1(labels, gt)

    # filter out data points with more than one label 
    single_idx = [i for i in range(len(gt)) if len(gt[i]) == 1]
    if len(single_idx) > 0:
        # there are points with only 1 label
        gt = np.asarray(gt, dtype=object).flatten()

        ari = metrics.adjusted_rand_score(gt[single_idx], labels[single_idx])
        nmi = metrics.normalized_mutual_info_score(gt[single_idx], labels[single_idx])
    else:
        # no points with only one label
        # ari & nmi are not defined
        ari = np.nan
        nmi = np.nan

    with open(outpath+"_fuzzycm.csv", "a") as f:
        f.write(";".join([str(k), str(seed), str(ari), str(nmi), str(pu), str(re), str(f1), str(fpc)]) + "\n")


def prf1(single, cats):
	alllbl = np.unique([x for y in cats for x in y])
	lblidx = [[l in x for x in cats] for l in alllbl]
	lblidx = np.array(lblidx, dtype=bool)

	goodlbl = Counter()

	purity = 0
	for i in np.unique(single):
		c = Counter([x for y in (single==i).nonzero()[0] for x in cats[y]])
		purity += c.most_common(1)[0][1]
		goodlbl[c.most_common(1)[0][0]] += 1
	purity /= len(single)
	f = np.array([Counter(single[l]).most_common(1)[0][1] / l.sum() for l in lblidx])
	recall = (lblidx.T * f).max(axis=1).sum() / len(single)
	return purity, recall, np.sqrt(purity * recall)


def estimate_cluster_sizes_with_hdbscan(files):
    for i, f in enumerate(files):
        print("File {}/16 (file: {})".format(i+1, f))

        data = load_data("./data/bws/{}_bws.csv.xz".format(f))
        data, _ = clean_data(data)

        outdir = "./eval/"

        outpath = outdir + f

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        X = get_normalized_vectors(data)
        gt = data["senseID"].values.tolist() # list of lists

        with open(outpath+"_est_ks_hdbs.csv", "w") as file:
            file.write(";".join(["k", "seed", "ari", "nmi", "pu", "re", "f1"])+ "\n")

        cluster_hdbscan(X, gt, outpath)


def cluster_with_estimated_ks(files, ks):
    seeds = list(range(1,4)) 

    for i, f in enumerate(files):
        print("File {}/16 (file: {})".format(i+1, f))

        data = load_data("./data/bws/{}_bws.csv.xz".format(f))
        data, _ = clean_data(data)

        outdir = "./eval/"

        outpath = outdir + f + "_est_ks"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        X = get_normalized_vectors(data)
        gt = data["senseID"].values.tolist() # list of lists

        ks_final = [ks[i]-x for x in range(10,31,10)] + [ks[i]+x for x in range(10,31,10)]

        with open(outpath+"_skm.csv", "w") as file:
            file.write(";".join(["k", "seed", "ari", "nmi", "pu", "re", "f1"]) + "\n")
        
        with open(outpath+"_fuzzycm.csv", "w") as file:
            file.write(";".join(["k", "seed", "ari", "nmi", "pu", "re", "f1", "fpc"])+ "\n")

        for ki, k in enumerate(ks_final):
            print("\t k: {}/6".format(ki+1))
            for s in seeds:
                cluster_spherical(X, k, s, gt, outpath)
                cluster_fuzzy_cmeans(X, k, s, gt, outpath)


def cluster(files):
    seeds = list(range(1,4)) 

    for i, f in enumerate(files):
        print("File {}/16 (file: {})".format(i+1, f))

        data = load_data("./data/bws/{}_bws.csv.xz".format(f))
        data, num_senses = clean_data(data)

        outdir = "./eval/"

        outpath = outdir + f

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        X = get_normalized_vectors(data)
        gt = data["senseID"].values.tolist() # list of lists

        ks_final = [num_senses-x for x in range(10,31,10)] + [num_senses+x for x in range(10,31,10)]

        if np.asarray(ks_final).max() > X.shape[0]:
            print("!!! RESET ks")
            # more senses than actual data points (probably due to diverse multi-labels)
            ks_final = [int(num_senses/2)-x for x in range(10,51,10)] + [int(num_senses/2)+x for x in range(10,51,10)]

        assert(np.asarray(ks_final).max() <= X.shape[0])
        assert(np.all(np.asarray(ks_final) > 1))

        with open(outpath+"_skm.csv", "w") as file:
            file.write(";".join(["k", "seed", "ari", "nmi", "pu", "re", "f1"]) + "\n")
        
        with open(outpath+"_fuzzycm.csv", "w") as file:
            file.write(";".join(["k", "seed", "ari", "nmi", "pu", "re", "f1", "fpc"])+ "\n")

        for ki, k in enumerate(ks_final):
            print("\t k: {}/6".format(ki+1))
            for s in seeds:
                cluster_spherical(X, k, s, gt, outpath)
                cluster_fuzzy_cmeans(X, k, s, gt, outpath)     


def view_results():
    for alg in ["hdbs", "skm", "fuzzycm"]:
        print("-"*5 + alg + "-"*5)
        path = "./eval/" # use your path
        all_files = glob.glob(path + "/*{}.csv".format(alg))

        l = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0, sep=";")
            df["file"] = filename.split("/")[-1].split("_" + alg + ".csv")[0]
            l.append(df)

        res = pd.concat(l, axis=0, ignore_index=True)
        print("Average performance:")
        print(res.groupby(["file", "k"]).mean().round(3))
        print("")
        print("Standard deviation:")
        print(res.groupby(["file", "k"]).std().round(3))
        print("")


if __name__ == "__main__":
    files_sorted_by_size = [
        'semeval2007task17', 
        'semeval2015task13',
        'semeval2013task12',
        'senseval3task1',
        'semeval2007task7',
        'senseval2',
        'senseval3task6_test',
        'senseval2_lexical_sample_test',
        'raganato_ALL',
        'senseval3task6_train',
        'senseval2_lexical_sample_train',
        'masc',
        'semcor',
        'wngt',
        'trainomatic',
        'omsti',
        ] # process smaller files first

    estimate_cluster_sizes_with_hdbscan(files_sorted_by_size)

    files = [
        'senseval2_lexical_sample_train',
        'senseval2_lexical_sample_test',
        'senseval3task6_train',
        'senseval3task6_test',
    ] # files on which HDBSCAN performed best

    num_clust = [118,83,106,77] # estimated cluster sizes (by HDBSCAN)


    cluster_with_estimated_ks(files, num_clust)
    cluster(files)

    view_results()
    

    

