from gensim.utils import simple_preprocess
from tqdm import tqdm
from datetime import datetime
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
import pickle
"""
Util class to go over training data
"""

class SentenceIterator:

    def __init__(self, filepath):
        self.filepath = filepath
        self.len = self.get_len()
        print(self.len)

    def __iter__(self):
        f = open(self.filepath, encoding="utf8")
        for i in tqdm(range(self.len)):
            yield simple_preprocess(f.readline(), deacc=False)

    def get_len(self):
        c = 0
        for line in tqdm(open(self.filepath, encoding="utf8")):
            c += 1
        return c

class TextFilePreprocessing:

    def __init__(self, filepath_in, filepath_out):
        self.filepath_in = filepath_in
        self.filepath_out = filepath_out
        self.len = self.get_len()

    def preprocess(self):
        f_in = open(self.filepath_in, encoding="utf8")
        lines = []
        f_out = open(self.filepath_out, "wb")
        for i in tqdm(range(self.len)):
            line = simple_preprocess(f_in.readline(), deacc=False)
            lines.append(line)
            if i % 5000000 == 0:
                pickle.dump(lines, f_out)
                lines = []
        pickle.dump(lines, f_out)
        f_out.close()

    def get_len(self):
        c = 0
        for line in tqdm(open(self.filepath_in, encoding="utf8")):
            c += 1
        return c

class PickleIterator:

    def __init__(self, filepath):
        self.filepath = filepath


    def loadall(self, filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def __iter__(self):
        for sentences in self.loadall(self.filepath):
            for i in tqdm(range(len(sentences))):
                yield sentences[i]

class EpochSaver(CallbackAny2Vec):

    '''Callback to save model after each epoch.'''


    def __init__(self, path_prefix):

        self.path_prefix = path_prefix


    def on_epoch_end(self, model):
        output_path = '{}.model'.format(self.path_prefix)
        model.save(output_path)

seconds_in_day = 24 * 60 * 60

class EpochLogger(CallbackAny2Vec):

    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        self.first_time = datetime.now()
        print("=======Epoch #{} start==============Start time {}=======".format(self.epoch, self.first_time))


    def on_epoch_end(self, model):
        self.later_time = datetime.now()
        difference = self.later_time - self.first_time

        print("=======Epoch #{} end==============End time {}==============Time taken: {}=======".format(self.epoch, self.later_time, difference))
        self.epoch += 1