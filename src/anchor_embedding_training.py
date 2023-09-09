import argparse
from tqdm import tqdm
import os
from anchors import AnchoredWord2Vec
from gensim.models import KeyedVectors
import multiprocessing
import pandas as pd
import numpy as np
from collections import defaultdict
from utils import SentenceIterator
import faiss


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


def get_index(vector_size):
    index = faiss.IndexFlatIP(vector_size)
    return index


def get_embeddings_as_array(w2v, normalize=True, top_n=-1, filter=None):
    if filter is None:
        filter = set()
    if top_n <= 0:
        top_n = len(w2v.index_to_key)
    res = list()

    index_to_key = list()
    i = 0
    for w in w2v.index_to_key:
        if i >= top_n:
            break
        if w in filter:
            continue

        if normalize:
            res.append(w2v[w]/np.linalg.norm(w2v[w]))
        else:
            res.append(w2v[w])
        index_to_key.append(w)
        i += 1

    res = np.array(res)
    return res, index_to_key


def get_nn(se, te, knn):
    index = get_index(se.shape[1])
    index.add(te)

    print('Running search query...')
    return index.search(se, knn)


def get_csls(se, te, knn, csls_knn):
    assert knn <= csls_knn

    st_distances, st_indices = get_nn(se, te, csls_knn)
    ts_distances, ts_indices = get_nn(te, se, csls_knn)

    res_distances = np.zeros((st_distances.shape[0], knn), dtype=st_distances.dtype)
    res_indices = np.zeros((st_distances.shape[0], knn), dtype=st_indices.dtype)

    print('Calculating CSLS...')

    rs_lst = [dists.mean() for dists in ts_distances]

    for sidx, sindices in enumerate(tqdm(st_indices)):
        rt = st_distances[sidx].mean()
        csls_dists = list()

        for ti, tidx in enumerate(sindices):
            rs = rs_lst[tidx]
            csls_dists.append(2*st_distances[sidx][ti] - rt - rs)

        csls_dists = np.array(csls_dists)
        max_idxs = np.argsort(csls_dists)[::-1][:knn]
        res_distances[sidx] = csls_dists[max_idxs]
        res_indices[sidx] = sindices[max_idxs]

    return res_distances, res_indices


def build_dico(src, tgt, src_index_to_key, tgt_index_to_key,
               top_n=1, csls_knn=10, min_similarity=-100, max_similarity=100):
    distances, indices = get_csls(src, tgt, top_n, csls_knn)
    res = dict()
    res_weights = dict()
    for i, idxs in enumerate(indices):
        sw = src_index_to_key[i]
        tmp_lst = [
            (tgt_index_to_key[idxs[j]], distances[i][j])
            for j in range(top_n)
            if min_similarity <= distances[i][j] <= max_similarity
        ]
        res[sw] = [item[0] for item in tmp_lst]
        res_weights[sw] = [item[1] for item in tmp_lst]

    return res, res_weights


def build_cycle_dico(src, tgt, src_index_to_key, tgt_index_to_key,
                     top_n=1, csls_knn=10, min_similarity=-100, max_similarity=100):
    # FIXME this is not the most efficient because it calculates CSLS twice
    s2t, s2t_weights = build_dico(
        src,
        tgt,
        src_index_to_key,
        tgt_index_to_key,
        top_n,
        csls_knn,
        min_similarity,
        max_similarity,
    )
    t2s, t2s_weights = build_dico(
        tgt,
        src,
        tgt_index_to_key,
        src_index_to_key,
        top_n,
        csls_knn,
        min_similarity,
        max_similarity,
    )

    res = dict()
    res_weights = dict()
    for k, v in s2t.items():
        tmp = list()
        tmp_weights = list()
        for idx, tw in enumerate(v):
            if k in t2s[tw]:
                tmp.append(tw)
                tmp_weights.append(s2t_weights[k][idx])
        if len(tmp) > 0:
            res[k] = tmp
            res_weights[k] = tmp_weights
    return res, res_weights


def dict2pd(dico, weights=None, columns=['src', 'tgt']):
    tmp = {
        columns[0]: [k for k in dico.keys() for _ in range(len(dico[k]))],
        columns[1]: [v for k in dico.keys() for v in dico[k]]
    }
    if weights is not None:
        tmp['weights'] = [w for k in dico.keys() for w in weights[k]]
    return pd.DataFrame.from_dict(tmp)


def anchor_training(vectors_src, training_corpus, bi_dict, size,
                    combined_vectors, limit, fixed, train_type,
                    max_final_vocab, min_count=3, epochs=5, vecs_to_copy=None):
    if combined_vectors:
        train_dict = get_combined_vectors_from_dict(
            bi_dict, vecs=vectors_src, fixed=fixed, n=limit)
        anchor_filename = f"data/anchor_words_{limit}_{'sg' if train_type else 'cbow'}.txt"
        with open(anchor_filename, 'w+') as file_anchors:
            for anchor_word in tqdm(list(train_dict.keys())):
                file_anchors.write(anchor_word+'\n')
    else:
        train_dict = get_dictionary_entries(
            bi_dict, vecs=vectors_src, fixed=fixed, n=limit)

    if vecs_to_copy is not None:
        for word in vecs_to_copy.index_to_key:
            if word not in train_dict:
                train_dict[word] = vecs_to_copy[word]

    print(f'Training anchor embeddings with {len(train_dict)} vectors')
    model = AnchoredWord2Vec(training_corpus, vector_size=size, window=5, min_count=min_count,
                     workers=multiprocessing.cpu_count(), sg=train_type, epochs=epochs,
                     fixed_vectors=train_dict, max_final_vocab=max_final_vocab)
    return model.wv


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
    parser.add_argument('fixed', type=int, help='Whether the anchor embeddings should be kept fixed', default=1)
    parser.add_argument('refine_it', type=int, help='Number if refining iterations', default=0)
    parser.add_argument('refine_top_n', type=int, help='Number of translations to look for in dico generation', default=3)
    parser.add_argument('--output_dir', type=str, help='Directory for aux output files', default=None)
    parser.add_argument('--epochs', type=int, help='Number of epochs for w2v training', default=5)
    parser.add_argument('--min_similarity', type=float, help='Minimum similarity for dictionay generation', default=-100.0)
    parser.add_argument('--max_similarity', type=float, help='Maximum similarity for dictionay generation', default=100.0)
    args = parser.parse_args()

    cwd = os.getcwd()  # Get the current working directory (cwd)
    os.chdir(cwd)

    fixed = args.fixed
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
    refine_it = args.refine_it
    top_n = args.refine_top_n
    num_epochs = args.epochs
    min_similarity = args.min_similarity
    max_similarity = args.max_similarity
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_corpus = SentenceIterator(tgt_data_path)
# Load a word2vec model stored in the C *text* format.
    vectors_src = KeyedVectors.load_word2vec_format(
        src_embedding_path, binary=False)
# vectors_src = KeyedVectors.load(src_embedding_path, mmap='r')

    colnames = ['src', 'tgt']
    bi_dict = pd.read_csv(bi_dict_path, names=colnames, sep=' ')
    model = anchor_training(vectors_src, training_corpus, bi_dict, size,
                            combined_vectors, limit, fixed, train_type,
                            max_final_vocab, epochs=num_epochs)

    for i in range(refine_it):
        print(f"Refining iteration {i+1}...")
        src_emb, src_index_to_key = get_embeddings_as_array(
            vectors_src,
            top_n=50000,
        )
        tgt_emb, tgt_index_to_key = get_embeddings_as_array(
            model,
            top_n=50000,
            filter=set(src_index_to_key),
        )
        new_bi_dict = dict2pd(
            *build_cycle_dico(
                src_emb,
                tgt_emb,
                src_index_to_key,
                tgt_index_to_key,
                top_n,
                min_similarity=min_similarity,
                max_similarity=max_similarity,
            ),
            colnames
        )
        # new_bi_dict = dict2pd(
        #     *build_dico(
        #         tgt_emb,
        #         src_emb,
        #         tgt_index_to_key,
        #         src_index_to_key,
        #         top_n,
        #         min_similarity,
        #         max_similarity,
        #     ),
        #     [colnames[1], colnames[0]]
        # )
        if output_dir is not None:
            new_bi_dict.to_csv(f'{output_dir}/bi_dict_{i+1}.txt', sep=' ', header=False, index=False)
        bi_dict = pd.concat([bi_dict, new_bi_dict], ignore_index=True)
        print(f'New dictionary size: {len(bi_dict)}')
        model = anchor_training(vectors_src, training_corpus, bi_dict, size,
                                combined_vectors, limit, fixed, train_type,
                                max_final_vocab, vecs_to_copy=model)

    print("Model finished, saving vectors...")
    model.save_word2vec_format(tgt_embedding_path)
    print("Done.")


if __name__ == "__main__":
    main()
