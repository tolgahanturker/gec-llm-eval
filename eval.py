from loader import Loader
import argparse
import globals
import os
import subprocess

def validate_arguments(args):
    # metric validation
    allowed_metrics = globals.CONFIG["EVALUATION"]["ALLOWED_METRICS"]
    if args.metric not in allowed_metrics:
        raise ValueError(f"Invalid metric: {args.metric}. Allowed values: {allowed_metrics}")
    
    # system_output_path validation
    if not os.path.exists(args.system_output_path):
        raise FileNotFoundError(f"File not found: {args.system_output_path}")

    # gold_data_path validation
    if not os.path.exists(args.gold_data_path):
        raise FileNotFoundError(f"File not found: {args.gold_data_path}")

    print("✓ All CLI arguments validated successfully.")

def parse_arguments():
    # parsing positional arguments
    parser = argparse.ArgumentParser(description = "Benchmarking LLMs for Grammar Error Correction (Evaluation)")
    parser.add_argument("metric", help = "name of the metric (options: m2scorer, errant, gleu)") # TODO: options to be updated based on CONFIG
    parser.add_argument("system_output_path", help = "path to the system output file")
    parser.add_argument("gold_data_path", help = "path to gold data file")
    args = parser.parse_args()

    # validating parsed arguments
    validate_arguments(args)

    # loading global parameters related to cli arguments
    globals.METRIC = args.metric
    globals.SYSTEM_OUTPUT_PATH = args.system_output_path
    globals.GOLD_DATA_PATH = args.gold_data_path

    # showing input settings
    if globals.CONFIG["GENERAL"]["DEBUG"]:
        print("############################################")
        print(f"Model\t\t\t: {globals.METRIC}")
        print(f"System Output Path\t: {globals.SYSTEM_OUTPUT_PATH}")
        print(f"Gold Data Path\t\t: {globals.GOLD_DATA_PATH}")
        print("############################################")

def evaluate():
    # only used for CoNLL2014 dataset
    if globals.METRIC == "m2scorer":
        command = [
            globals.CONFIG["EVALUATION"]["PYTHON2_FULL_PATH_FOR_M2SCORER"],
            globals.CONFIG["EVALUATION"]["M2SCORER_PATH"],
            globals.SYSTEM_OUTPUT_PATH,
            globals.GOLD_DATA_PATH
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred. Error code: {e.returncode}")

def main():
    # loading yaml for configuration parameters
    Loader.load_config("config.yaml")

    # parsing cli arguments
    parse_arguments()

    # performing evaluation
    evaluate()

# python eval.py m2scorer "results/gpt-5.1-2025-11-13_official-2014.combined-withalt.m2_zero-shot_neutral.txt_202512200016.txt" "data/test/conll2014/official-2014.combined-withalt.m2"
if __name__ == "__main__":
    main()