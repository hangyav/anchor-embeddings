# Python program to generate word vectors using Word2Vec
from gensim import utils
from gensim.models import Word2Vec
import sys
from pathlib import Path
# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')


class Corpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, path) -> None:
        self.path = path
        self.i = 0

    def __iter__(self):
        for line in open(self.path, "rt", encoding="utf-8"):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line.replace('\n', ' '))
            self.i += 1


def generate_embeddings(path):
    corpus = Corpus(path)
    model = Word2Vec(min_count=10, workers=-1, negative=25,
                     vector_size=200, window=8, sample=1e-4)
    model.build_vocab(corpus)
    print(model)
    model.train(corpus, total_examples=model.corpus_count,
                epochs=15, compute_loss=True)
    return model


def main():
    wiki_dump_file_in = Path(sys.argv[1])
    print(wiki_dump_file_in)
    model_file_out = f'models/{wiki_dump_file_in.stem}.model'
    print(f'Training embeddings on {wiki_dump_file_in}...')

    model = generate_embeddings(path=wiki_dump_file_in)

    model.save(model_file_out)

    print(f'Successfully saved embedding model to {model_file_out}...')


if __name__ == '__main__':
    main()
