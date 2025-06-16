import yaml
import globals
import argparse
from data_loader import DataLoader
from api_caller import APICaller
import os
from pathlib import Path
import errant
import spacy

def load_config(path):
    # reading configuration parameters in config.yaml and storing it in global parameter CONFIG
    with open(path, "r") as file:
        globals.CONFIG = yaml.safe_load(file)

def parse_arguments():
    # parsing arguments provided from run command
    parser = argparse.ArgumentParser(description = "gec-llm-eval")
    parser.add_argument("path", help = "path to the M2 data file")
    parser.add_argument("model", help = "name of the LLM to use (options: gpt-4o)")
    parser.add_argument("experimentMode", help = "experiment mode (options: zero-shot or few-shot)")
    parser.add_argument("-pid", "--promptId", type = int, help = "prompt id (...)", default = 1)
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
        print("####################")

def load_data():
    data = DataLoader.load_data()

    # writing data info
    if globals.CONFIG["GENERAL"]["DEBUG"]:
        for item in data:
            print(f"Sentence ID\t: {item["sentence_id"]}")
            print(f"Test Sentence\t: {item["test_sentence"]}")
            print()
    
    return data

def write_results(results):
    # for ERRANT compatible tokenization of hypothesis result
    nlp = spacy.load('en_core_web_sm')

    directory = "results"
    os.makedirs(directory, exist_ok = True)

    # writing api result
    path_result_api = os.path.join(directory, f"result_api_{globals.MODEL}_{Path(globals.INPUT_FILE_PATH).stem}_{globals.EXPERIMENT_MODE}_{globals.PROMPT_ID}.txt")
    with open(path_result_api, "w", encoding="utf-8") as file:
        for item in results:
            file.write(str(item) + "\n")
    
    # writing raw result
    path_result_raw = os.path.join(directory, f"result_raw_{globals.MODEL}_{Path(globals.INPUT_FILE_PATH).stem}_{globals.EXPERIMENT_MODE}_{globals.PROMPT_ID}.txt")
    with open(path_result_raw, "w", encoding="utf-8") as file:
        for item in results:
            # tokenizing corrected_sentence using spaCy to ensure compatibility
            doc = nlp(item["corrected_sentence"])
            tokens = [token.text for token in doc]
            output = " ".join(tokens)
            file.write(output + "\n")
            
    # writing m2 result
    path_result_m2 = os.path.join(directory, f"result_m2_{globals.MODEL}_{Path(globals.INPUT_FILE_PATH).stem}_{globals.EXPERIMENT_MODE}_{globals.PROMPT_ID}.m2")
    annotator = errant.load('en', nlp)
    with open(path_result_m2, "w", encoding="utf-8") as file:
        for item in results:
            test_sentence = annotator.parse(item["test_sentence"])
            file.write("S " + test_sentence.text + "\n")
            corrected_sentence = annotator.parse(item["corrected_sentence"], tokenise = True) # tokenization is needed because corrected_sentence is not tokenized
            edits = annotator.annotate(test_sentence, corrected_sentence)
            for edit in edits:
                edit_output = edit.to_m2(id=0)
                file.write(edit_output + "\n")
            file.write("\n")

def main():

    # loading global configuration
    load_config("config.yaml")

    # parsing arguments
    parse_arguments()

    # loading input data
    data = load_data()

    # llm API calls
    result = APICaller.run(data)

    # preprocessing for evaluation
    write_results(result)
    
    print("finish")

# python3 main.py 'data/bea2019/test/ABCN.test.bea19.orig' gpt-4o
# python.exe .\main.py 'data\jfleg\test\test.src' "gpt-4o" "zero-shot" 2
if __name__ == "__main__":
    main()