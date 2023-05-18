from gensim.models import KeyedVectors
import sys 
from tqdm import tqdm

src_path = sys.argv[1]
src2_path = sys.argv[2] 
word_len = 2

f = open(src2_path, encoding="utf8")
file_data = f.readlines()
header = file_data[0]
file_data = file_data[1:]

old_file_data = open(src_path, encoding='utf8').readlines()
header_old = old_file_data[0]
old_file_data = old_file_data[1:]

extract_arg = lambda x, y: int(x.split(' ')[y])

total_length = extract_arg(header, 0) + extract_arg(header_old, 0)
vec_size = extract_arg(header, 1)

with open(src_path, 'w', encoding='utf-8') as out_f:
    out_f.write(f'{total_length} {vec_size}\n')
    for s in tqdm(old_file_data):
        out_f.write(s)

out_f.close()

with open(src_path, 'a', encoding='utf-8') as out_f:
    for s in tqdm(file_data):
        out_f.write(s)


f.close()