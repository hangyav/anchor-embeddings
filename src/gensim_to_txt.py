from gensim.models import Word2Vec
import sys

src_path = sys.argv[1]
out_path = sys.argv[2] 

print(f'Loading "{src_path}"...')
model = Word2Vec.load(src_path)
print(f'Saving word vectors at "{out_path}"...')
model.wv.save_word2vec_format(out_path)
print('Saved vectors successfully!')