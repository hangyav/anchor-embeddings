import json
import subprocess
import os
import sys


def run_command(command, args):
    formatted_command = command.format(**args)
    print(f"Running command:\n{formatted_command}\n")
    process = subprocess.Popen(formatted_command, shell=True)
    process.wait()
    if process.returncode != 0:
        raise Exception('Error in subprocess!')


def run_experiment(experiment, experiment_index):
    project_dir = os.getcwd()  # get current directory
    src_dir = os.path.join(project_dir, 'src')  # append 'src' to the path

    out_name = experiment['tgt_emb_name'] if bool(experiment['save_tgt_embeddings_without_concat']) else experiment['tgt_model_new_name']

    commands = {
        "find_identical_words": f"python3 {src_dir}/find_identical_words.py {{src_model_file}} {{tgt_model_file}} {{output_identical_word_pair_file}} {{top_vocab}}",
        "anchor_embedding_training": f"python3 {src_dir}/anchor_embedding_training.py {{output_identical_word_pair_file}} {{tgt_model_training_data}} {out_name} 0 {{src_model_file}} {{embedding_type_cbow_or_sg}} {{vector_dim}} {{vector_count}} {{fixed}}",
        "concat_kv": f"python3 {src_dir}/concat_kv.py {out_name} {{src_model_file}} {{tgt_model_new_name}}"
    }
    print(
        f"\nExperiment {experiment_index + 1} parameters:\n{json.dumps(experiment, indent=4)}\n")

    for step in ["find_identical_words", "anchor_embedding_training", "concat_kv"]:
        print(
            f"\nRunning step '{step}' for experiment {experiment_index + 1}\n")
        run_command(commands[step], experiment)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <experiment_config.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file) as f:
        data = json.load(f)

    for i, experiment in enumerate(data['experiments']):
        print(f"\nStarting experiment {i + 1}...\n")
        run_experiment(experiment, i)

    project_dir = os.getcwd()  # get current directory
    evaluate_command = f"python3 {project_dir}/MUSE/evaluate.py --tgt_emb {{tgt_model_new_name}} --src_emb {{src_model_file}} --tgt_lang {{tgt_lang}} --src_lang {{src_lang}} --dico_eval {{path_to_evaluation_file}} --exp_name {{experiment_name}} --cuda {{cuda}}"
    print(
        f"\nRunning final evaluation with the following parameters:\n{json.dumps(data['evaluate'], indent=4)}\n")
    run_command(evaluate_command, data['evaluate'])


if __name__ == "__main__":
    main()
