from gensim.models import Word2Vec as OriginalWord2Vec

class AnchoredWord2Vec(OriginalWord2Vec):
    def __init__(self, *args, fixed_vectors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_vectors = fixed_vectors

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
