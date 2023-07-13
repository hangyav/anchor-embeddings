from gensim.models import KeyedVectors
import sys
from tqdm import tqdm

src_path = sys.argv[1]
tgt_path = sys.argv[2]
out_path = sys.argv[3]
top_vocab = -1 if len(sys.argv) < 5 else int(sys.argv[4])
word_len = 2

print('Reading files...')
wv_from_text_src = KeyedVectors.load_word2vec_format(src_path, binary=False)
print('Source file read successfully!')
print('Reading target...')
wv_from_text_tgt = KeyedVectors.load_word2vec_format(tgt_path, binary=False)
print('Target file read successfully!')

print('Computing intersection...!')
candidates = set(wv_from_text_src.index_to_key)
tgt_vocab = wv_from_text_tgt.index_to_key
if top_vocab > 0:
    tgt_vocab = tgt_vocab[:top_vocab]
identical_words = [
    word
    for word in tgt_vocab
    if word in candidates
]
identical_words = [word for word in identical_words if len(word) > int(word_len)]
# identical_words = set(wv_from_text_src.index_to_key).intersection(wv_from_text_tgt.index_to_key)
# identical_words = {word for word in identical_words if len(word) > int(word_len)}
print(f'Success! Total identical words {len(identical_words)}.')

print(f'Writing results to file "{out_path}"')
with open(out_path, 'w', encoding='utf-8') as out_f:
    for word in tqdm(identical_words):
        out_f.write(word + ' ' + word + '\n')
