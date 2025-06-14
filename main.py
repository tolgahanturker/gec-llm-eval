import yaml
import globals
import argparse
from data_loader import DataLoader

def load_config(path):
    # reading configuration parameters in config.yaml and storing it in global parameter CONFIG
    with open(path, "r") as f:
        globals.CONFIG = yaml.safe_load(f)

def parse_arguments():
    # parsing arguments provided from run command
    parser = argparse.ArgumentParser(description = "gec-llm-eval")
    parser.add_argument("path", help = "path to the M2 data file")
    parser.add_argument("model", help = "name of the LLM to use (...)")
    parser.add_argument("-em", "--experimentMode", type = int, help = "experiment mode (...)", default = 0)
    parser.add_argument("-pid", "--promptId", type = int, help = "prompt id (...)", default = 0)
    args = parser.parse_args()

    # loading global config parameters related to arguments
    globals.INPUT_FILE_PATH = args.path
    globals.MODEL = args.model
    globals.EXPERIMENT_MODE = args.experimentMode
    globals.PROMPT_ID = args.promptId

    # writing arguments
    if globals.CONFIG["GENERAL"]["DEBUG"]:
        print(f"m2 data file\t: {args.path}")
        print(f"model\t\t: {args.model}")
        print(f"exp. mode\t: {args.experimentMode}")
        print(f"prompt id\t: {args.promptId}")

def load_data():
    data = DataLoader.load_data()

    # writing data info
    if globals.CONFIG["GENERAL"]["DEBUG"]:
        for item in data:
            print(f"Sentence ID\t: {item["sentence_id"]}")
            print(f"Test Sentence\t: {item["test_sentence"]}")
            print()

def main():

    # loading global configuration
    load_config("config.yaml")

    # parsing arguments
    parse_arguments()

    # loading input data
    data = load_data()

    print("finish")


# python3 main.py 'data/bea2019/test/ABCN.test.bea19.orig' gpt-4o
if __name__ == "__main__":
    main()