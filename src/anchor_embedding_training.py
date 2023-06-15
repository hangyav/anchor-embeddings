import argparse
from tqdm import tqdm
import os
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
import pandas as pd
import numpy as np
from collections import defaultdict
from unidecode import unidecode
from utils import SentenceIterator


def get_dictionary_entries(bi_dict, vecs, fixed=True, n=0):
    train_dict = dict()
    train_dict['__fix_training'] = fixed
    train_dict['__fix_init'] = True
    if n == 0:
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
    not_found = 0
    train_dict = defaultdict(list)
    if n == 0:
        limit = len(bi_dict)
    else:
        limit = n
    for i, row in bi_dict.iterrows():
        decoded_de = str(row['tgt'])
        decoded_en = str(row['src'])
        if decoded_en not in vecs:
            not_found += 1
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


def main():
    parser = argparse.ArgumentParser(
        description='Train anchor-embeddings using gensim library.')
    parser.add_argument(
        'bi_dict_path', help='Path to your bilingual dictionary.')
    parser.add_argument('tgt_data_path', help='Path to your training data.')
    parser.add_argument('tgt_embedding_path', help='Path to your output file.')
    parser.add_argument('limit', type=int,
                        help='Max number of anchor vectors.')
    parser.add_argument(
        'src_embedding_path', help='Path to the source embedding for the anchor vectors.')
    parser.add_argument(
        'train_type', help='Type of training. Use "sg" for skip-gram, anything else for CBOW.')
    parser.add_argument('size', type=int, help='Dimension of the vectors.')
    parser.add_argument('max_final_vocab', type=int,
                        help='Maximum final vocabulary size.')
    args = parser.parse_args()

    cwd = os.getcwd()  # Get the current working directory (cwd)
    os.chdir(cwd)

    fixed = True
    combined_vectors = True
    print("Train embeddings start.")

    bi_dict_path = os.path.abspath(cwd+'/'+args.bi_dict_path)
    tgt_data_path = os.path.abspath(cwd+'/'+args.tgt_data_path)
    tgt_embedding_path = os.path.abspath(cwd+'/'+args.tgt_embedding_path)
    limit = args.limit
    src_embedding_path = os.path.abspath(cwd+'/'+args.src_embedding_path)
    train_type = 1 if args.train_type.lower().strip() == 'sg' else 0
    size = args.size
    max_final_vocab = args.max_final_vocab

    training_corpus = SentenceIterator(tgt_data_path)
# Load a word2vec model stored in the C *text* format.
    vectors_src = KeyedVectors.load_word2vec_format(
        src_embedding_path, binary=False)
# vectors_src = KeyedVectors.load(src_embedding_path, mmap='r')

    colnames = ['src', 'tgt']
    bi_dict = pd.read_csv(bi_dict_path, names=colnames, sep=' ')
    if combined_vectors:
        train_dict = get_combined_vectors_from_dict(
            bi_dict, vecs=vectors_src, fixed=fixed, n=limit)
        anchor_filename = f"data/anchor_words_{limit}_{args.train_type.lower().strip()}.txt"
        with open(anchor_filename, 'w+') as file_anchors:
            for anchor_word in tqdm(list(train_dict.keys())):
                file_anchors.write(anchor_word+'\n')
    else:
        train_dict = get_dictionary_entries(
            bi_dict, vecs=vectors_src, fixed=fixed, n=limit)

    model = Word2Vec(training_corpus, vector_size=size, window=5, min_count=3,
                     workers=multiprocessing.cpu_count(), sg=train_type, epochs=5,
                     fixed_vectors=train_dict, max_final_vocab=max_final_vocab)

    print("Model finished, saving vectors...")
    model.wv.save_word2vec_format(tgt_embedding_path)
    print("Done.")


if __name__ == "__main__":
    main()
