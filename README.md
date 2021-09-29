# BERTifiedWordSenses

BERTifiedWordSenses (BWS) is a dense high-dimensional clustering dataset derived from [UFSAC](https://github.com/getalp/UFSAC). BWS provides a total of over 2.8M sense-annotated words within sentences along with 768-dimensional contextual word embeddings extracted from [BERT](https://huggingface.co/transformers/model_doc/bert.html#bert).

## Download
The full BWS dataset is available for download [here](https://tu-dortmund.sciebo.de/s/VusSNXhjt1BDgCC).

## Usage

1) Install the necessary dependencies from `requirements.txt`.
1) To use BWS directly, download the dataset from [here](https://tu-dortmund.sciebo.de/s/VusSNXhjt1BDgCC), and place the files under `./data`.
1) To recreate the experiments described in our paper, run `cluster_analysis.py`. 
1) To view the clustering results, run `view_results.py`. You can pass a different input directory (e.g., `./eval_paper/eval`) by using the `-p` flag.  
1) To re-generate the BWS dataset:
    1) Download [UFSAC v2.1](https://github.com/getalp/UFSAC/blob/master/corpus/ufsac-public-2.1.link.txt) and place it under `./ufsac`.
    1) Run `load_ufsac.py`.
    1) Run `bertify_ufsac.py`.

For using BWS in a project, we suggest using our `load_data()` and `clean_data()` functions from `load_bws.py` to load the data into a pandas DataFrame.

## Example
BWS files are provided as compressed CSV files (`.csv.xz`) and have the following columns:

- sentID	
- sentence	
- word	
- pos	(part of speech)
- startIDX	(of the word within the sentence)
- endIDX	
- wn_30_sense	
- num_senses	
- senseID	
- avg (768-dimensional word embedding vector <-- averaged token embeddings)

**Note: The `.csv.xz` files contain one column per vector dimension. We provide the `clean_data()` function in `load_bws.py` for merging them into a single column of a pandas DataFrame.**

## Attribution

Please credit our work by citing the introductory paper

Gloria Feher, Erich Schubert  
BERTified Word Senses: A Dense High-Dimensional Clustering Dataset  
under review, 2021
