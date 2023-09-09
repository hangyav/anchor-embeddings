# Multilingual Embeddings Experiments

This project allows you to run a series of experiments for multilingual embeddings using a single JSON configuration file.

## Setup

Before running the experiments, you need to set up a virtual environment and install the required packages. Here's how you can do it:

```bash
python3 -m venv venv
source venv/bin/activate
pip -V  # ensure correct pip location
pip install -r requirements.txt
pip install faiss-cpu # optionally for speedup on cpu or faiss-gpu to run on GPU
# put MUSE under the ./MUSE directory
```


## Running the Experiments

To run the experiments, use the following command:

```bash
python3 run_experiment.py <experiment_config.json>
```

or to save the output log and results to a file

```bash
python3 run_experiment.py <experiment_config.json> 2>&1 | tee <log_file>
```

Replace <experiment_config.json> with the path to your JSON configuration file. For more information regarding the JSON configuration files see the documentation under the `experiments` directory.

The script will run each experiment in the order they are defined in the JSON file. It will print a message before starting each experiment, before running each step, and before running the final evaluation. The messages include the experiment number, the step name, and the parameters for the step or evaluation. 

## Troubleshooting

If you encounter any issues while running the experiments, here are a few things you can try:

- Ensure that all the paths in the JSON configuration file are correct and that the files exist.

- Make sure that you have the necessary permissions to read the files and write to the directories specified in the JSON configuration file.

- Check that the `Word2Vec` class in the `gensim` package has a `fixed_vectors` parameter. If it doesn't, you might need to update the `gensim` package or modify the `word2vec.py` file.

- If you're getting out-of-memory errors, try reducing the `vector_count` or using a machine with more memory.

If you're still having issues, please open an issue on the project's GitHub page or contact the project maintainers.

## Contributing

Contributions to this project are welcome! If you have a feature request, bug report, or proposal, please open an issue on the project's (future) GitHub page. If you want to contribute code, please open a pull request.

Please make sure to follow the project's code style and write tests for any new features or changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
