from gensim.models import KeyedVectors
import sys 
from tqdm import tqdm

src_path = sys.argv[1]
n = int(sys.argv[2])
out_path = sys.argv[3]

print(f'Reading file and retrieving top {n} vectors...')
wv_from_text_src = KeyedVectors.load_word2vec_format(src_path, binary=False, limit=n)
print(f'File processed successfully! Vectors retrieved: {len(wv_from_text_src)}')


print('Saving...')
wv_from_text_src.save_word2vec_format(out_path)

print('Success!')




