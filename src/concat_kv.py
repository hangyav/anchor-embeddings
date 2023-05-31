import argparse
from tqdm import tqdm

def concatenate_files(src1_path, src2_path, output_path):
    with open(src2_path, encoding="utf8") as f:
        file_data = f.readlines()
    header = file_data[0]
    file_data = file_data[1:]

    with open(src1_path, encoding='utf8') as f:
        old_file_data = f.readlines()
    header_old = old_file_data[0]
    old_file_data = old_file_data[1:]

    extract_arg = lambda x, y: int(x.split(' ')[y])

    total_length = extract_arg(header, 0) + extract_arg(header_old, 0)
    vec_size = extract_arg(header, 1)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f'{total_length} {vec_size}\n')
        for s in tqdm(old_file_data):
            out_f.write(s)

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for s in tqdm(file_data):
            out_f.write(s)

def main():
    parser = argparse.ArgumentParser(description='Concatenate two KeyedVectors files.')
    parser.add_argument('src1_path', help='Path to the first source file.')
    parser.add_argument('src2_path', help='Path to the second source file.')
    parser.add_argument('output_path', help='Path to the output file.')
    args = parser.parse_args()

    concatenate_files(args.src1_path, args.src2_path, args.output_path)

if __name__ == "__main__":
    main()
