from loader import Loader
import argparse
import globals
import os
from api_caller import APICaller
from reporter import Reporter

def validate_arguments(args):
    # model validation
    allowed_models = globals.CONFIG["GENERAL"]["ALLOWED_LLMS"]
    if args.model not in allowed_models:
        raise ValueError(f"Invalid model: {args.model}. Allowed values: {allowed_models}")

    # data_path validation
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"File not found: {args.data_path}")

    # prompt_path validation
    if not os.path.exists(args.prompt_path):
        raise FileNotFoundError(f"File not found: {args.prompt_path}")

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

def main():
    # loading yaml for configuration parameters
    Loader.load_config("config.yaml")

    # parsing cli arguments
    parse_arguments()

    # loading input data
    data = Loader.load_data()

    # llm API calls
    result, llm_output_path = APICaller.run(data)
    
    # writing results for evaluation
    Reporter.write_results_for_eval(result, llm_output_path)

<<<<<<< HEAD

# run script: python main.py gpt-5.1-2025-11-13 "data/test/deneme" "instructions/zero-shot_neutral.txt"
=======
# python3 main.py 'data/bea2019/test/ABCN.test.bea19.orig' gpt-4o
# python.exe .\main.py 'data\jfleg\test\test.src' "gpt-4o" "zero-shot" 2
>>>>>>> 66e96e2705675ab848f28ca87d030528cd94e218
if __name__ == "__main__":
    main()