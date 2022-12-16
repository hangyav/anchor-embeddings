import sys
import os

from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.test.utils import datapath
import multiprocessing
import pandas as pd
import numpy as np
from collections import defaultdict
from unidecode import unidecode
from utils import SentenceIterator

"""
This script is an example on how to utilize the gensim library to train anchor-embeddings.
Usage: Provide the source embedding for the anchor vectors under en_embedding_path
Set variables fixed and combined_vectors to the mode you want to train in.

Call the script as python anchor_embedding_training.py DICTIONARY_PATH TRAINING_DATA_PATH SAVE_PATH NUM_ANCHORS EN_EMBEDDING_PATH
where the three first arguments are files for your bilingual dictionary, training data and output file,
while NUM_ANCHORS is an integer representing the max number of anchor vectors.
"""

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
os.chdir(cwd)

sys.path.append('gensim/')
print("Files in %r: %s" % (os.getcwd(), files))



def get_dictionary_entries(bi_dict, vecs, fixed=True, n=0):
    train_dict = dict()
    train_dict['__fix_training'] = fixed
    train_dict['__fix_init'] = True
    if n == 0:
        # return full dict
        limit = len(bi_dict)
    else:
        limit = n
    for i, row in bi_dict.iterrows():
        if len(train_dict) >= limit:
            break
        decoded_de = str(row['tgt'])
        decoded_en = str(row['src'])
        if decoded_de not in train_dict:
            if decoded_en in vecs:
                train_dict[decoded_de] = vecs[decoded_en]
    return train_dict


def get_combined_vectors_from_dict(bi_dict, vecs, fixed=True, n=0):
    not_found=0
    train_dict = defaultdict(list)
    if n == 0:
        limit = len(bi_dict)
    else:
        limit = n
    for i, row in bi_dict.iterrows():
      
        decoded_de = str(row['tgt'])
        decoded_en = str(row['src'])
        if decoded_en not in vecs:
            not_found+=1
            continue
        elif len(train_dict) >= limit and decoded_de not in train_dict:
            break
        else:
            train_dict[decoded_de].append(vecs[decoded_en])
    combined_dict = dict()
    combined_dict['__fix_training'] = fixed
    combined_dict['__fix_init'] = True
    for entry, vectors in train_dict.items():
        combined_dict[entry] = np.mean(vectors, axis=0)

 
    
    return combined_dict

fixed = True
combined_vectors = True
print("Train embeddings start.")
bi_dict_path = os.path.abspath(cwd+'/'+sys.argv[1]) #"data/lexicons/mk-en_lexicon_no_test.txt"
tgt_data_path = os.path.abspath(cwd+'/'+sys.argv[2])  #"data/macedonian/mkwiki.full"
tgt_embedding_path = os.path.abspath(cwd+'/'+sys.argv[3])  #"data/macedonian/embeddings/W2V_MK_300D_CBOW_aligned.kv"
limit = int(sys.argv[4])
src_embedding_path = os.path.abspath(cwd+'/'+sys.argv[5])  #"data/embeddings/EN_300D_CBOW.kv"
train_type = 1 if sys.argv[6].lower().strip() =='sg' else 0 
size=int(sys.argv[7])

training_corpus = SentenceIterator(tgt_data_path)
# Load a word2vec model stored in the C *text* format.
vectors_src = KeyedVectors.load_word2vec_format(datapath(src_embedding_path), binary=False) 
#vectors_src = KeyedVectors.load(src_embedding_path, mmap='r')

colnames = ['src','tgt']
bi_dict = pd.read_csv(bi_dict_path, names=colnames, sep=' ')
from tqdm import tqdm
if combined_vectors:
    train_dict = get_combined_vectors_from_dict(bi_dict, vecs=vectors_src, fixed=fixed, n=limit)
    #save the anchor words. Maybe to a txt file   
    anchor_filename = f"data/anchor_words_{limit}_{sys.argv[6].lower().strip()}.txt"
    with open(anchor_filename,'w+') as file_anchors:
        for anchor_word in tqdm(list(train_dict.keys())):
            file_anchors.write(anchor_word+'\n')
else:
    train_dict = get_dictionary_entries(bi_dict, vecs=vectors_src, fixed=fixed, n=limit)

#TO DO: Change size to 300
model = Word2Vec(training_corpus, vector_size=size, window=5, min_count=3,
                 workers=multiprocessing.cpu_count(), sg=train_type, epochs=5,
                 fixed_vectors=train_dict)

print("Model finished, saving vectors...")
model.wv.save_word2vec_format(tgt_embedding_path)
print("Done.")
