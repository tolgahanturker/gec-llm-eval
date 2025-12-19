import subprocess

def validate_arguments(args):
    

    # metric validation
    allowed_metrics = globals.CONFIG["GENERAL"]["ALLOWED_EVAL_METRICS"]
    if args.metric not in allowed_metrics:
        raise ValueError(f"Invalid metric: {args.metric}. Allowed values: {allowed_metrics}")

    print("✓ All CLI arguments validated successfully.")

def parse_arguments():
    # parsing positional arguments
    parser = argparse.ArgumentParser(description = "Benchmarking LLMs for Grammar Error Correction")
    parser.add_argument("model", help = "name of the model (options: gpt-4o)") # TODO: options to be updated based on CONFIG
    parser.add_argument("data_path", help = "path to the input data file")
    parser.add_argument("prompt_path", help = "path to the prompt file")
    args = parser.parse_args()

    # validating parsed arguments
    validate_arguments(args)

    # loading global parameters related to cli arguments
    globals.MODEL = args.model
    globals.DATA_PATH = args.data_path
    globals.PROMPT_PATH = args.prompt_path

    # showing input settings
    if globals.CONFIG["GENERAL"]["DEBUG"]:
        print("############################################")
        print(f"Model\t\t\t: {globals.MODEL}")
        print(f"Data Path\t\t: {globals.DATA_PATH}")
        print(f"Prompt Path\t\t: {globals.PROMPT_PATH}")
        print("############################################")

def evaluate(result_path):

    if globals.EVAL_METRIC == "m2scorer":
        command = [
            globals.CONFIG["GENERAL"]["PYTHON2_FULL_PATH_FOR_M2SCORER"],
            "./data/conll2014/m2scorer/scripts/m2scorer.py",
            result_path,
            "./data/conll2014/alt/official-2014.combined-withalt_test.m2"
        ]

        try:
            subprocess.run(command, check=True)
        except e:
            print(f"An error occured. Error code: {e.returncode}")

def main():

    # loading yaml for configuration parameters
    Loader.load_config("config.yaml")

    # performing evaluation
    evaluate(result_path)

    return

# run script: python eval.py m2scorer "data/test/deneme" "instructions/zero-shot_neutral.txt"
if __name__ == "__main__":
    main()






