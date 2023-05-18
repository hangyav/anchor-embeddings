from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from gensim.test.utils import datapath

import sys
from pathlib import Path


def main():
    model_file_in = sys.argv[1]
    print(model_file_in)
    model = KeyedVectors.load_word2vec_format(
        model_file_in, binary=True, unicode_errors='ignore')
    print('10 most similar words to flughafen:')
    print(model.most_similar('flughafen', topn=10))
    print('10 most similar words to apfel:')
    print(model.most_similar('apfel', topn=10))
    print('10 most similar words to apple:')
    print(model.most_similar('apple', topn=10))
    print('10 most similar words to london:')
    print(model.most_similar('london', topn=10))
    print('10 most similar words to politiker:')
    print(model.most_similar('politiker', topn=10))
    print("model.similarity('reagan', 'thatcher'):", model.similarity('reagan', 'thatcher'))
    print("model.similarity('banana', 'apple'):",model.similarity('banana', 'apple'))
    print("model.similarity('berlin', 'london'):",model.similarity('berlin', 'london'))
    print("model.similarity('mathematics', 'john'):",model.similarity('mathematics', 'john'))
    print("model.similarity('mathematics', 'mathematics'):",model.similarity('mathematics', 'mathematics'))


if __name__ == '__main__':
    main()
