# Multilingual Embeddings Experiments

This project allows you to run a series of experiments for multilingual embeddings using a single JSON configuration file.

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
            "tgt_model_new_name": "models/tl_en_model_200k.txt",
            "embedding_type_cbow_or_sg": "cbow",
            "vector_dim": "300",
            "vector_count": "200000"
        },
        {
            "src_model_file": "models/tl_en_model_200k.txt",
            "tgt_model_file": "models/hil_model.txt",
            "output_identical_word_pair_file": "data/dictionaries/en_tl_hil_identicals_from_200k.txt",
            "tgt_model_training_data": "data/hil_literary_religious_preprocessed",
            "tgt_model_new_name": "models/hil_tl_en_model.txt",
            "embedding_type_cbow_or_sg": "cbow",
            "vector_dim": "300",
            "vector_count": "200000"
        }
    ],
    "evaluate": {
        "tgt_model_new_name": "models/hil_tl_en_model.txt",
        "src_model_file": "models/en_model_200k.txt",
        "tgt_lang": "en",
        "src_lang": "hil",
        "path_to_evaluation_file": "data/dictionaries/En-Hil Lexicon/en-hil_TEST_first_200.txt",
        "experiment_name": "python_script_test",
        "cuda": "False"
    }
}
```
## Running the Experiments
To run the experiments, use the following command:

```bash
python3 run_experiment.py <experiment_config.json>
```
Replace <experiment_config.json> with the path to your JSON configuration file.

The script will run each experiment in the order they are defined in the JSON file. It will print a message before starting each experiment, before running each step, and before running the final evaluation. The messages include the experiment number, the step name, and the parameters for the step or evaluation.

