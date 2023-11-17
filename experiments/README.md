# Experiments

## JSON Configuration

The experiments are defined in a JSON configuration file. The `evaluate` step is
defined once at the end of the JSON file. To only evaluate, leave the
`experiments` step empty (`[]`).

### Examples

* Supervised bilingual English-Hiligaynon: [en-hil.supervised.json](en-hil.supervised.json)
* Supervised multilingual English-Spanish-Indonesian-Hiligaynon: [en-es-id-hil.supervised.json](en-es-id-hil.supervised.json)
* Weakly-supervised (identical words) multilingual English-Hiligaynon: [en-hil.identical.json](en-hil.identical.json)

For building the necessary initial monolingual word embeddings see [gensim](https://radimrehurek.com/gensim/models/word2vec.html),
[word2vec](https://github.com/dav/word2vec),
[fasttext](https://github.com/facebookresearch/fastText), etc. For further details
related to data resources, see [data](../data).

## Arguments

Here's a brief explanation of each argument in the JSON configuration file:

### Experiments

- `src_model_file`: The source language word embeddings in text format.

- `tgt_model_file`: The target language word embeddings. Optional: only in case
of the weakly-supervised setup.

- `output_identical_word_pair_file`: The output file to save identical word
pairs for later analysis. Optional: only in case of the weakly-supervised setup.

- `tgt_model_training_data`: Target language corpus.

- `tgt_model_new_name`: The output file for the intermediate embeddings.

- `embedding_type_cbow_or_sg`: The type of embeddings (cbow or sg).

- `vector_dim`: The dimension of the vectors.

- `vector_count`: The number most frequent word vectors to save.

- `save_tgt_embeddings`: If set to `True`, the target embeddings will be saved
independently. This is used for evaluation. See next.

- `tgt_emb_name`: Path of the item related to `save_tgt_embeddings`.

- `fixed`: Whether to keep anchor embeddings fixed.

- `top_vocab`: Consider only this many most frequent words for identical pair
finding. Optional.

- `refine_it`: Number of dictionary refinement iterations.

- `refine_top_n`: Number of translations for each word during refinement.

- `train_dico`: List of training dictionaries for supervised training. Optional.

### Evaluation

- `src_model_file`: The source language word embeddings.

- `tgt_model_new_name`: The final target language word embeddings.

- `src_lang`: The source language ISO.

- `tgt_lang`: The target language ISO.

- `path_to_evaluation_file`: Test dictionary.

- `experiment_name`: The name of the experiment for logging.

- `cuda`: If set to `True`, CUDA will be used for the evaluation.
