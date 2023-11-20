# Anchored bi- and multilingual word embeddings

This project aims at building cross-lingual word embeddings for low-resource
languages, which lack large amounts of monolingual data. Instead of building
monolingual word embeddings for multiple languages and aligning them in two
independent steps, it builds the target language embeddings in a single step by
*anchoring* them to the embeddings space of a high resource language. Both
bilingual and multilingual embeddings are supported. For further details see
our published [papers](#papers).

![anchor](https://github.com/hangyav/anchor-embeddings/assets/414596/75f59783-5114-47af-a56e-8c774c4d91a7)

## Setup

```bash
pip install -r requirements.txt
# put MUSE under the ./MUSE directory
```

NOTE: Developed with python version 3.8.18.


## Running the Experiments

To run the experiments, use the following command:

```bash
python3 run_experiment.py <experiment_config.json>
```

or to save the output log and results to a file

```bash
python3 run_experiment.py <experiment_config.json> 2>&1 | tee <log_file>
```

Replace `<experiment_config.json>` with the path to your JSON configuration file.
For more information regarding the JSON configuration files see the
documentation under the [experiments](experiments) directory.

JSON configuraions can be built manually, or generated using
`build_chain_setup.py`. For further details see

```
python3 build_chain_setup.py -h
```

## Troubleshooting

If you encounter any issues while running the experiments, here are a few things you can try:

- Ensure that all the paths in the JSON configuration file are correct and that the files exist.

- Make sure that you have the necessary permissions to read the files and write to the directories specified in the JSON configuration file.

- If you're getting out-of-memory errors, try reducing the `vector_count` or using a machine with more memory.

If you're still having issues, please open an issue on the project's GitHub page or contact the project maintainers.

## Papers
[1] Viktor Hangya, Silvia Severini, Radoslav Ralev, Alexander Fraser and Hinrich Schütze. 2023. [Multilingual Word Embeddings for Low-Resource Languages using Anchors and a Chain of Related Languages](#). In Proceedings of the The 3nd Workshop on Multi-lingual Representation Learning (MRL)

[2] Tobias Eder, Viktor Hangya, and Alexander Fraser. 2021. [Anchor-based Bilingual Word Embeddings for Low-Resource Languages](https://aclanthology.org/2021.acl-short.30). In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)

[3] Leah Michel, Viktor Hangya, and Alexander Fraser. 2020. [Exploring Bilingual Word Embeddings for Hiligaynon](https://aclanthology.org/2020.lrec-1.313.pdf), a Low-Resource Language. In Proceedings of The 12th Language Resources and Evaluation Conference


```
@inproceedings{hangya-etal-2023-multilingual-anchor,
    author = {Hangya, Viktor and Severini, Silvia and Ralev, Radoslav and Fraser, Alexander and Sch{\"u}tze, Hinrich},
    title = {{Multilingual Word Embeddings for Low-Resource Languages using Anchors and a Chain of Related Languages}},
    booktitle = {Proceedings of the The 3nd Workshop on Multi-lingual Representation Learning (MRL)},
    year = {2023},
}

@inproceedings{eder-etal-2021-anchor,
    title = {"Anchor-based Bilingual Word Embeddings for Low-Resource Languages"},
    author = "Eder, Tobias  and Hangya, Viktor  and Fraser, Alexander",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-short.30",
    pages = "227--232",
}

@inproceedings{michel2020exploring,
  title={Exploring bilingual word embeddings for Hiligaynon, a low-resource language},
  author={Michel, Leah and Hangya, Viktor and Fraser, Alexander},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={2573--2580},
  url = {https://aclanthology.org/2020.lrec-1.313.pdf},
  year={2020}
}
```

## Contributing

Contributions to this project are welcome! If you have a feature request, bug
report, or proposal, please open a new issue. If you want to contribute code,
please open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more
details.
