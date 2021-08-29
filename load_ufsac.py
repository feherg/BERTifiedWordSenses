import pandas as pd
from bs4 import BeautifulSoup
import os


if not os.path.exists("./ufsac"):
    os.makedirs("./ufsac")
    print("Please download UFSAC 2.1 from this URL and place it under ./ufsac:")
    print("https://drive.google.com/file/d/1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO")
    print("To remove duplicates, please remove all raganato files except raganato_ALL.xml")
    sys.exit(0)


allfiles = os.listdir("./ufsac")

num_sense_annotations_all = 0

for idx_f,f in enumerate(allfiles[5:9]):
    print("Processing file {}/16".format(idx_f+1))
    sentid = [] # prefix for xml file + sentence idx
    sent = []
    word = []
    start = [] # start idx in sentence
    end = [] # end idx in sentence
    wn30 = [] # wn30_key from ufsac
    pos = [] # part of speech annotation

    with open("./ufsac/"+f) as fp:

        soup = BeautifulSoup(fp, "xml")

        # go through sentences
        for idx_s, s in enumerate(soup.find_all("sentence")):
            curr_sent = ""
            curr_sentid = f + "_" + str(idx_s)
            # get all words
            words = list(s.find_all("word"))

            num_sense_annotations = 0

            for idx_w, w in enumerate(words):
                w_str = w.get("surface_form")
                w_wn = w.get("wn30_key")
                if w_str is not None and w_wn is not None: 
                    num_sense_annotations += 1
                    # there is a sense annotation
                    # get indices in string
                    curr_sent_len = len(curr_sent)
                    start.append(curr_sent_len)
                    end.append(curr_sent_len + len(w_str))

                    wn30.append(w_wn)
                    pos.append(w.get("pos"))
                    
                    w_lem = w.get("lemma")

                    # if available, add lemma instead of surface form
                    if w_lem is not None:
                        word.append(w_lem)
                    else:
                        word.append(w_str)

                # add word to current sent
                curr_sent += w_str + " "
                 
                
            
            curr_sent = curr_sent.strip() # remove extra whitespace

            sent.extend([curr_sent]*num_sense_annotations)
            sentid.extend([curr_sentid]*num_sense_annotations)

            num_sense_annotations_all += num_sense_annotations


    data = pd.DataFrame({"sentID": sentid, "sentence": sent, "word": word, "pos":pos, "startIDX": start, "endIDX": end, "wn_30_sense": wn30})

    if not os.path.exists("./data/ufsac_concat"):
        os.makedirs("./data/ufsac_concat")

    data.to_csv("./data/ufsac_concat/ufsac_sentences_{}.csv.xz".format(idx_f), index=False)

    del data

print("Number of all sense annotated words: ",num_sense_annotations_all)