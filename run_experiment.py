import json
import subprocess
import os
import sys

PYTHON = sys.executable

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

    experiment.setdefault('refine_it', 0)
    experiment.setdefault('refine_top_n', 3)
    experiment.setdefault('epochs', 5)
    experiment.setdefault('min_sim', -100)
    experiment.setdefault('max_sim', 100)
    experiment.setdefault('anchor_output_dir', f"{experiment['tgt_model_new_name']}.aux")

    commands = {
        "find_identical_words": f"{PYTHON} {src_dir}/find_identical_words.py {{src_model_file}} {{tgt_model_file}} {{output_identical_word_pair_file}} {{top_vocab}}",
        "anchor_embedding_training": f"{PYTHON} {src_dir}/anchor_embedding_training.py {{output_identical_word_pair_file}} {{tgt_model_training_data}} {out_name} 0 {{src_model_file}} {{embedding_type_cbow_or_sg}} {{vector_dim}} {{vector_count}} {{fixed}} {{refine_it}} {{refine_top_n}} --output_dir {{anchor_output_dir}} --epochs {{epochs}} --min_similarity {{min_sim}} --max_similarity {{max_sim}}",
        "concat_kv": f"{PYTHON} {src_dir}/concat_kv.py {out_name} {{src_model_file}} {{tgt_model_new_name}}"
    }
    print(
        f"\nExperiment {experiment_index + 1} parameters:\n{json.dumps(experiment, indent=4)}\n")

    if 'train_dico' in experiment:
        if type(experiment['train_dico']) == list:
            if not os.path.exists(experiment['anchor_output_dir']):
                os.makedirs(experiment['anchor_output_dir'])

            path = os.path.join(experiment['anchor_output_dir'], 'train.dico')
            with open(path, 'w') as fout:
                for dico in experiment['train_dico']:
                    with open(dico) as fin:
                        print(fin.read().strip(), file=fout)

            experiment['train_dico'] = path
        steps = ["anchor_embedding_training", "concat_kv"]
        experiment['output_identical_word_pair_file'] = experiment['train_dico']
    else:
        steps = ["find_identical_words", "anchor_embedding_training", "concat_kv"]
    for step in steps:
        print(
            f"\nRunning step '{step}' for experiment {experiment_index + 1}\n")
        run_command(commands[step], experiment)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {PYTHON} script.py <experiment_config.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file) as f:
        data = json.load(f)

    for i, experiment in enumerate(data['experiments']):
        print(f"\nStarting experiment {i + 1}...\n")
        run_experiment(experiment, i)

    if 'evaluate' in data:
        project_dir = os.getcwd()  # get current directory
        evaluate_command = f"{PYTHON} {project_dir}/MUSE/evaluate.py --tgt_emb {{tgt_model_new_name}} --src_emb {{src_model_file}} --tgt_lang {{tgt_lang}} --src_lang {{src_lang}} --dico_eval {{path_to_evaluation_file}} --exp_name {{experiment_name}} --cuda {{cuda}}"
        print(
            f"\nRunning final evaluation with the following parameters:\n{json.dumps(data['evaluate'], indent=4)}\n")
        run_command(evaluate_command, data['evaluate'])


if __name__ == "__main__":
    main()
