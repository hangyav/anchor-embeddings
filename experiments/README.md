# Experiments

## JSON Configuration

The experiments are defined in a JSON configuration file. Each experiment is a JSON object that includes the parameters for the `find_identical_words`, `anchor_embedding_training`, and `concat_kv` steps. The `evaluate` step is defined once at the end of the JSON file.

Here's an example of what the JSON configuration file might look like:

```json
{
  "experiments": [
    {
      "src_model_file": "models/en_model_200k.txt",
      "tgt_model_file": "models/tl_model_200k.txt",
      "output_identical_word_pair_file": "data/dictionaries/en_tl_identicals_from_200k.txt",
      "tgt_model_training_data": "data/tlwiki-latest-pages-articles_preprocessed.txt",
      "tgt_model_new_name": "models/tl_en_model_400k.txt",
      "embedding_type_cbow_or_sg": "cbow",
      "vector_dim": "300",
      "vector_count": "200000",
      "save_tgt_embeddings_without_concat": false, //optional: WARNING: Setting this will save the embeddings twice, once concatenated and once only the target which can take up a lot of space if done on each step.
      "tgt_emb_name": "", // optional and dependent on "save_tgt_embeddings_without_concat":
      "top_vocab": -1,
      "fixed": 1

    },
    {
      "src_model_file": "models/tl_en_model_400k.txt",
      "tgt_model_file": "models/hil_model.txt",
      "output_identical_word_pair_file": "data/dictionaries/en_tl_hil_identicals_from_200k.txt",
      "tgt_model_training_data": "data/hil_literary_religious_preprocessed",
      "tgt_model_new_name": "models/hil_tl_en_model_600k.txt",
      "embedding_type_cbow_or_sg": "cbow",
      "vector_dim": "300",
      "vector_count": "200000",
      "save_tgt_embeddings_without_concat": true,
      "tgt_emb_name": "models/hil_emb_anchors.txt", // necessary if "save_tgt_embeddings_without_concat" is set
      "top_vocab": -1,
      "fixed": 1
    }
  ],
  "evaluate": {
    "tgt_model_new_name": "models/hil_emb_anchors.txt", // carefully select the correct embeddings file
    "src_model_file": "models/en_model_200k.txt",
    "tgt_lang": "en",
    "src_lang": "hil",
    "path_to_evaluation_file": "data/dictionaries/En-Hil Lexicon/en-hil_TEST_first_200.txt",
    "experiment_name": "python_script_test",
    "cuda": "False"
  }
}
```

## Arguments

Here's a brief explanation of each argument in the JSON configuration file:

- `src_model_file`: The source model file for the `find_identical_words`, `anchor_embedding_training`, and `concat_kv` steps.

- `tgt_model_file`: The target model file for the `find_identical_words` step.

- `output_identical_word_pair_file`: The output file for the `find_identical_words` step and input file for the `anchor_embedding_training` step.

- `tgt_model_training_data`: The training data for the `anchor_embedding_training` step.

- `tgt_model_new_name`: The output file for the `anchor_embedding_training` and `concat_kv` steps, and the target embeddings for the `evaluate` step.

- `embedding_type_cbow_or_sg`: The type of embeddings (CBOW or skip-gram) for the `anchor_embedding_training` step.

- `vector_dim`: The dimension of the vectors for the `anchor_embedding_training` step.

- `vector_count`: The number of vectors for the `anchor_embedding_training` step.

- `save_tgt_embeddings`: If set to `True`, the target embeddings will be saved without concatenation after the `anchor_embedding_training` step.

- `tgt_lang`: The target language for the `evaluate` step.

- `src_lang`: The source language for the `evaluate` step.

- `path_to_evaluation_file`: The path to the evaluation file for the `evaluate` step.

- `experiment_name`: The name of the experiment for the `evaluate` step.

- `cuda`: If set to `True`, CUDA will be used for the `evaluate` step.
