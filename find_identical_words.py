from gensim.models import KeyedVectors
import sys 
from tqdm import tqdm

src_path = sys.argv[1]
tgt_path = sys.argv[2] 
out_path = sys.argv[3]
word_len = sys.argv[4]

print('Reading files...')
wv_from_text_src = KeyedVectors.load_word2vec_format(src_path, binary=False)
print('Source file read successfully!')
print('Reading target...')
wv_from_text_tgt = KeyedVectors.load_word2vec_format(tgt_path, binary=False)
print('Target file read successfully!')

print('Computing intersection...!')
identical_words = set(wv_from_text_src.index_to_key).intersection(wv_from_text_tgt.index_to_key)
identical_words = {word for word in identical_words if len(word) > int(word_len)}
print(f'Success! Total identical words {len(identical_words)}.')

print(f'Writing results to file "{out_path}"')
with open(out_path, 'w', encoding='utf-8') as out_f:
    for word in tqdm(identical_words):
        out_f.write(word + ' ' + word + '\n')




