from transformers import BertTokenizerFast, TFBertModel
import tensorflow as tf
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os
from itertools import chain


##### Functions #####
def string_to_encoded_idx(s, tokenizer):
    string_idx2token_idx = {}
    tok_idx = 1 # first token is [CLS]
    for m in re.finditer(r'\W|\w\.[\w\.]?[\w\.]?|\w+[.]?\w+|\w', s):
        # NOTE: Translation for the regex above:
        # \W = non-alphanumeric char OR
        # \w\.[\w\.]?[\w\.]? = alphanumeric . with possible 2x repetition (--> D.C. , J.F.K. , ....) OR
        # \w+[.]?\w+ = alphanumeric optional . and some further alphanumeric ( so we get the final full stop as a single char) OR
        # \w = single alphanumeric (--> a, I, ...) 

        # find start, end indices and word
        start, end, word = m.start(), m.end(), m.group()

        if len(re.findall(r'\s',word)) > 0:
            # the regex above also matches whitespaces, so ignore them
            continue

        # tokenize word and map indices to tokens
        tok = tokenizer.tokenize(word)

        string_idx2token_idx[(start, end)] = [tok_idx, tok_idx+len(tok)]
        tok_idx += len(tok)


    return string_idx2token_idx


def combine_encoded_indices_to_words(string_idx2token_idx, start, end):
    str_idx = string_idx2token_idx.keys()

    try:
        key_start = [i for i in str_idx if i[0] == start][0]
    except IndexError as e:
        key_start = [i for i in str_idx if i[0] == start+1 or i[0] == start-1][0] # check for off by 1 error / inconsistency in parsing (e.g., a spacebar counted into the word)
    
    try:
        key_end = [i for i in str_idx if i[1] == end][0]
    except IndexError as e:
        key_end = [i for i in str_idx if i[1] == end+1 or i[1] == end-1][0] # check for off by 1 error / inconsistency in parsing (e.g., a spacebar counted into the word)

    enc_start = string_idx2token_idx[key_start][0]
    enc_end = string_idx2token_idx[key_end][1]

    return enc_start, enc_end


def process_dataset_masked_output(data, model, tokenizer, outpath):
    hidden_dim = model.config.get_config_dict("bert-base-cased")[0]["hidden_size"]

    # process entire dataset
    averaged = []
    remove = []

    num_too_long = 0

    for index, dat in tqdm(data.iterrows()):

        if isinstance(dat["sentence"], float):
            print("Sent is NaN")
            # sentence is NaN
            # no sentence, stop processing
            remove.append(True)
            averaged.append([0]*hidden_dim)
            continue

        string_idx2token_idx = string_to_encoded_idx(dat["sentence"], tokenizer)

        try:
            start_enc, end_enc = combine_encoded_indices_to_words(string_idx2token_idx, dat["startIDX"], dat["endIDX"])
        except:
            # some sentences are simply incorrectly parsed, or not even complete - in that case they should be removed
            # combine_encoded_indices_to_words() already checks for off-by-1 errors / inconsistencies from the original parsing
            print("NO MATCH")
            remove.append(True)
            averaged.append([0]*hidden_dim)
            continue

        inputs = tokenizer(dat["sentence"], return_tensors="tf")
        if inputs["input_ids"].shape[1] > 512:
            # skip triple
            remove.append(True)
            averaged.append([0]*hidden_dim)
            num_too_long += 1
            continue

        outputs = model(inputs)

        last_hidden_states = outputs.last_hidden_state.numpy()

        # mask out everything except for the current word
        mask = np.zeros(last_hidden_states.shape[1], dtype=bool) # mask rows

        if start_enc == end_enc:
            mask[start_enc] = True
        else:
            mask[start_enc : end_enc] = True

        hidden_states_no_ents = last_hidden_states[:,mask]


        # combine hidden states 
        if hidden_states_no_ents.shape[1] < 1:
            # don't append
            averaged.append([0]*hidden_dim)
            remove.append(True)
        elif hidden_states_no_ents.shape[1] == 1:
            # simply append, there's only one token embedding
            averaged.append(hidden_states_no_ents.flatten())
            remove.append(False)
        else:
            # there are > 1 token embeddings -> average them
            hidden_states_no_ents_avg = hidden_states_no_ents.mean(axis=1)
            averaged.append(hidden_states_no_ents_avg)
            remove.append(False)
        

    print("Triples flagged for removal due to length > 512: ", num_too_long)
    
    averaged = pd.DataFrame(np.vstack(averaged))
    data = pd.concat([data, averaged], axis=1) # save each value in the 768-dim vector as one col in the dataframe
    data["remove"] = remove

    # clean
    data = data[data["remove"]==False]

    senses = data["wn_30_sense"]
    senses_split = senses.str.split(';', expand=False) # split senses into list
    data["num_senses"] = senses_split.apply(lambda x: len(x))
    data["wn_30_sense"] = senses_split

    # unique senses in the dataset
    unique = set(list(chain.from_iterable(data["wn_30_sense"].values.tolist())))

    # create a dict mapping unique senses to a numerical id
    unique = list(unique)
    sense2id = dict(zip(unique, range(len(unique))))
    data["senseID"] = data["wn_30_sense"].apply((lambda x: [sense2id[i] for i in x]))

    # save
    data.to_csv(outpath, index=False)


if __name__ == "__main__":

    ##### BERT #####

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = TFBertModel.from_pretrained('bert-base-cased')

    ##### I/O #####

    outdir = "./data/bws/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for idx_f in range(2,3):
        print("File {}/16".format(idx_f+1))

        data = pd.read_csv("./data/ufsac_concat/ufsac_sentences_{}.csv.xz".format(idx_f))
        fname = data["sentID"].iloc[0].split(".xml")[0]

        outfile = fname + "_bws.csv.xz"

        outpath = outdir + outfile

        process_dataset_masked_output(data, model, tokenizer, outpath)



