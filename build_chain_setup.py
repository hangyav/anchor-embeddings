import argparse
import json
import os


def main(
    languages,
    output_file,
    corpora,
    embeddings,
    output_dir,
    eval_dict=None,
    embedding_type="cbow",
    dim=300,
    vector_count=200000,
    eval_s2t=False,
    cuda=False,
    fixed=True,
    top_vocab=-1,
    train_dict=None,
    accumulative_train_dico=False,
    refine_it=0,
    refine_top_n=3,
    min_sim=-100.0,
    max_sim=100,
):
    assert len(languages) > 1
    res = dict()

    exps = list()
    prev_emb = embeddings.format(lang=languages[0])
    for idx in range(1, len(languages)):
        lang = languages[idx]
        tgt_emb = embeddings.format(lang=lang)
        corpus = corpora.format(lang=lang)
        save_without_concat = (idx == len(languages) - 1)
        tgt_emb_name = ''
        if save_without_concat:
            tgt_emb_name = os.path.join(output_dir, f'{lang}_model_final.txt')
        output_identical_word_pair_file = os.path.join(
            output_dir,
            f'{"_".join(languages[:idx+1])}_identical_word_pairs.txt',
        )
        tgt_model_new_name = os.path.join(
            output_dir,
            f'{"_".join(languages[:idx+1])}_model_chain.txt',
        )

        exps.append({
            'src_model_file': prev_emb,
            'tgt_model_file': tgt_emb,
            'output_identical_word_pair_file': output_identical_word_pair_file,
            'tgt_model_training_data': corpus,
            'tgt_model_new_name': tgt_model_new_name,
            'embedding_type_cbow_or_sg': embedding_type,
            'vector_dim': dim,
            'vector_count': vector_count,
            'save_tgt_embeddings_without_concat': save_without_concat,
            'tgt_emb_name': tgt_emb_name,
            'fixed': 1 if fixed else 0,
            'top_vocab': top_vocab,
            'refine_it': refine_it,
            'refine_top_n': refine_top_n,
            'min_sim': min_sim,
            'max_sim': max_sim,
        })
        if train_dict is not None:
            if accumulative_train_dico:
                exps[-1]['train_dico'] = [
                    train_dict.format(src=languages[si], tgt=lang)
                    for si in range(idx)
                ]
            else:
                exps[-1]['train_dico'] = train_dict.format(src=languages[idx-1], tgt=lang)
        prev_emb = tgt_model_new_name

    res['experiments'] = exps

    if eval_dict is not None:
        eval = {
            'tgt_model_new_name': exps[-1]['tgt_emb_name'] if eval_s2t else exps[0]['src_model_file'],
            'src_model_file': exps[0]['src_model_file'] if eval_s2t else exps[-1]['tgt_emb_name'],
            'tgt_lang': languages[-1] if eval_s2t else languages[0],
            'src_lang': languages[0] if eval_s2t else languages[-1],
            'path_to_evaluation_file': eval_dict,
            'experiment_name': f'{"_".join(languages)}_chain',
            'cuda': 'True' if cuda else 'False',
        }
        res['evaluate'] = eval

    with open(output_file, 'w') as fout:
        json.dump(res, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", help="List of languages in the chain in order", type=str, nargs='+', required=True)
    parser.add_argument("--output_file", help="Output json file.", type=str, required=True)
    parser.add_argument("--corpora", help="Text file path containing language placeholder. Eg: ./{lang}.wiki.txt", type=str, required=True)
    parser.add_argument("--embeddings", help="Embedding path containing language placeholder. Eg: ./{lang}.model.txt", type=str, required=True)
    parser.add_argument("--output_dir", help="Output directory for the training.", type=str, required=True)
    parser.add_argument("--embedding_type", help="cbow or sg", type=str, choices={"cbow", "sg"}, default="cbow")
    parser.add_argument("--dim", help="Vector dimension", type=int, default=300)
    parser.add_argument("--vector_count", help="Top frequent embeddings to keep", type=int, default=200000)
    parser.add_argument("--train_dict", help="Supervised dictionary with placeholders. Eg: ./{src}-{tgt}.tsv", type=str, default=None)
    parser.add_argument("--accumulative_train_dico", help="Use all languages' dico in the source space", type=int, default=0)
    parser.add_argument("--eval_dict", help="Path to final evaluation dictionary.", type=str, default=None)
    parser.add_argument("--eval_s2t", help="Final eval should be source to target?", type=int, default=0)
    parser.add_argument("--fixed", help="Whether anchors should be fixed during training", type=int, default=1)
    parser.add_argument("--top_vocab", help="Number of most frequent words from the target to consider for identical word pair search", type=int, default=-1)
    parser.add_argument("--refine_it", help="Number of refinement iterations", type=int, default=0)
    parser.add_argument("--refine_top_n", help="top n for dictionay induction", type=int, default=3)
    parser.add_argument("--min_sim", help="minimum similarity for dictionay induction", type=float, default=-100.0)
    parser.add_argument("--max_sim", help="maximum similarity for dictionay induction", type=float, default=100.0)

    args = parser.parse_args()
    main(**vars(args))
