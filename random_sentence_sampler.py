import random
import sys
from tqdm import tqdm
from gensim.utils import simple_preprocess

"""
Helper script to randomly sample a smaller amount of text from a larger file.
"""

def random_sampler(filename):
    sample = []
    total_nr_tokens = 0
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        random_set = sorted(random.sample(range(filesize), int(5e6)))
        i = 0
        with tqdm(total=int(5e6)) as pbar:
            while total_nr_tokens < 5e6:
                f.seek(random_set[i])
                f.readline()
                line = f.readline().rstrip()
                nr_tokens_in_line = len(simple_preprocess(line.decode('utf8')))
                total_nr_tokens += nr_tokens_in_line
                sample.append(line)
                i +=1
                pbar.update(nr_tokens_in_line)
    return sample, total_nr_tokens

filename = sys.argv[1]
outfile = sys.argv[2]

print('Sampling file...')
sample, total_nr_tokens = random_sampler(filename)
print(f'Sample of size {total_nr_tokens} collected!')
with open(outfile, 'wb') as fout:
    for line in sample:
        fout.write(line)
        fout.write(b'\n')

