import sys
import itertools


def read_dico(file, reverse=False):
    res = dict()
    if reverse:
        src = 1
        tgt = 0
    else:
        src = 0
        tgt = 1

    with open(file) as f:
        for line in f:
            data = line.strip().split()
            if len(data) < max(src, tgt) + 1:
                print(f'Ignoreing line: {line}')
                continue
            res.setdefault(data[src], set()).add(data[tgt])

    return res


def pivot_dico(dico1, dico2):
    res = dict()

    src_words = set(dico1.keys()) | set(dico2.keys())
    for src_word in src_words:
        translations1 = dico1.get(src_word, set())
        translations2 = dico2.get(src_word, set())
        res[src_word] = list(itertools.product(translations1, translations2))

    return res


if __name__ == '__main__':
    dico1_path = sys.argv[1]
    dico2_path = sys.argv[2]
    out_path = sys.argv[3]
    reverse_dico = False if len(sys.argv) < 5 else int(sys.argv[4])
    out_sep = ' ' if len(sys.argv) < 6 else sys.argv[5]

    dico1 = read_dico(dico1_path, reverse_dico)
    dict2 = read_dico(dico2_path, reverse_dico)

    pivoted_dico = pivot_dico(dico1, dict2)

    with open(out_path, 'w') as f:
        for src_word, translations in pivoted_dico.items():
            for translation in translations:
                print(f'{translation[0]}{out_sep}{translation[1]}', file=f)
