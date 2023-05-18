from gensim.models import Word2Vec 
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath
import multiprocessing
from utils import PickleIterator, SentenceIterator, EpochSaver, EpochLogger
import sys
from tqdm import tqdm



"""
Standard training of a word embedding without the proposed method.
""" 

src_data_path = sys.argv[1]
w2v_embedding_path = sys.argv[2]

from os.path import exists


sentences = SentenceIterator(src_data_path)

if exists(w2v_embedding_path):
    print('Loaded model ', w2v_embedding_path)
    model = Word2Vec.load(w2v_embedding_path)
    print("Training model on %s" % (src_data_path))
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[EpochSaver(w2v_embedding_path), EpochLogger()], report_delay=1)
else:
    model = Word2Vec(sg=0, vector_size=300, window=5, min_count=3, epochs=5, workers=multiprocessing.cpu_count())
    model.build_vocab(corpus_iterable=sentences)
    print("Training model on %s" % (src_data_path))
    model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[EpochSaver(w2v_embedding_path), EpochLogger()])

print("Model finished, saving...")
model.save(w2v_embedding_path)
print("Done.")
