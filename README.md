# BERTifiedWordSenses

BERTifiedWordSenses (BWS) is a dense high-dimensional clustering dataset derived from [UFSAC](https://github.com/getalp/UFSAC). BWS provides a total of over 2.8M sense-annotated words within sentences along with 768-dimensional contextual word embeddings extracted from [BERT](https://huggingface.co/transformers/model_doc/bert.html#bert).

## Usage

1) Install the necessary dependencies from `requirements.txt`.
1) Download [UFSAC v2.1](https://github.com/getalp/UFSAC/blob/master/corpus/ufsac-public-2.1.link.txt) and place it under `./ufsac`.
1) To use BWS, download the dataset from [here](https://tu-dortmund.sciebo.de/s/VusSNXhjt1BDgCC), and place the files under `./data`.
1) To re-generate the BWS dataset, first run `load_ufsac.py`, and then `bertify_ufsac.py`.
1) To recreate the experiments described in our paper, run `cluster_analysis.py`. 

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

## Attribution

Please credit our work by citing the introductory paper

Gloria Feher, Erich Schubert  
BERTified Word Senses: A Dense High-Dimensional Clustering Dataset  
under review, 2021
