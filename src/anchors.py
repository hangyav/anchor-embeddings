from gensim import utils
from gensim.models import Word2Vec as OriginalWord2Vec
from gensim.models.fasttext import FastText, FastTextKeyedVectors
from gensim.models.fasttext_inner import MAX_WORDS_IN_BATCH
import logging
import numpy as np
from numpy import float32 as REAL
from numpy import ones

logger = logging.getLogger(__name__)

class AnchoredWord2Vec(OriginalWord2Vec):
    def __init__(self, *args, fixed_vectors=None, **kwargs):
        self.fixed_vectors = fixed_vectors
        super().__init__(*args, **kwargs)

    def init_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.wv.resize_vectors(seed=self.seed)

        vocab_len = len(self.wv.vectors)
        self.wv.vectors_lockf = np.ones(vocab_len, dtype=REAL)
        if hasattr(self.wv, 'vectors_vocab'):
            ft_model = True
        else:
            ft_model = False

        if bool(self.fixed_vectors):
            for key, idx in self.wv.key_to_index.items():
                if key in self.fixed_vectors:
                    self.wv.vectors[idx] = self.fixed_vectors[key]
                    if ft_model:
                        self.wv.vectors_vocab[idx] = self.fixed_vectors[key]
                    if self.fixed_vectors['__fix_training']:
                        self.wv.vectors_lockf[idx] = 0.0

        if ft_model:
            self.wv.vectors_vocab_lockf = self.wv.vectors_lockf

        if self.hs:
            self.syn1 = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)

class AnchoredFastTextKeyedVectors(FastTextKeyedVectors):
    def __init__(self, *args, fixed_vectors=None, **kwargs):
        self.fixed_vectors = fixed_vectors
        super().__init__(*args, **kwargs)

    def adjust_vectors(self):
        """Adjust the vectors for words in the vocabulary.

        The adjustment composes the trained full-word-token vectors with
        the vectors of the subword ngrams, matching the Facebook reference
        implementation behavior.

        ### Made adjustments so that we pass over fixed vectors for subword updates
        """
        if self.bucket == 0:
            self.vectors = self.vectors_vocab  # no ngrams influence
            return

        self.vectors = self.vectors_vocab[:].copy()
        for i, key in enumerate(self.index_to_key):
            if bool(self.fixed_vectors):
                if key in self.fixed_vectors:
                    if self.fixed_vectors['__fix_init']:
                        continue
            ngram_buckets = self.buckets_word[i]
            for nh in ngram_buckets:
                self.vectors[i] += self.vectors_ngrams[nh]
            self.vectors[i] /= len(ngram_buckets) + 1


class AnchoredFastText(AnchoredWord2Vec, FastText):
    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 max_final_vocab=None, fixed_vectors=None):
        self.load = utils.call_on_class_only
        self.load_fasttext_format = utils.call_on_class_only
        self.callbacks = callbacks
        if word_ngrams != 1:
            raise NotImplementedError("Gensim's FastText implementation does not yet support word_ngrams != 1.")
        self.word_ngrams = word_ngrams
        if max_n < min_n:
            # with no eligible char-ngram lengths, no buckets need be allocated
            bucket = 0

        self.wv = AnchoredFastTextKeyedVectors(vector_size, min_n, max_n, bucket, fixed_vectors=fixed_vectors)
        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as desired after any vocab growth
        self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)

        super().__init__(
            sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=vector_size, epochs=epochs,
            callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window,
            max_vocab_size=max_vocab_size, max_final_vocab=max_final_vocab,
            min_count=min_count, sample=sample, sorted_vocab=sorted_vocab,
            null_word=null_word, ns_exponent=ns_exponent, hashfxn=hashfxn,
            seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, fixed_vectors=fixed_vectors)
