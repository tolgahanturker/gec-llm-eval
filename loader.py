import yaml
import globals
from pathlib import Path
from tokenizer import Tokenizer

class Loader:

    @staticmethod
    def load_config(path):
        # reading configuration parameters in config.yaml
        with open(path, "r") as file:
            # loading global CONFIG parameter with config.yaml content
            globals.CONFIG = yaml.safe_load(file)
    
    @staticmethod
    def load_data():
        sent_id = 0
        data = []

        # to load m2 formatted data
        if globals.DATA_PATH.endswith(".m2"):
            with Path(globals.DATA_PATH).open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if line.startswith("S "):
                        input_sentence = line[2:]

                        # Davis' detokenization heuristic with Moses Detokenizer
                        token_list = input_sentence.split()
                        input_sentence = Tokenizer.davis_detokenize(token_list)

                        sent_id += 1
                        data.append(dict(sentence_id = sent_id, test_sentence = input_sentence))
        # to load txt formatted data
        else:
            with Path(globals.DATA_PATH).open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    # Davis' detokenization heuristic with Moses Detokenizer
                    token_list = line.split()
                    line = Tokenizer.davis_detokenize(token_list)

                    sent_id += 1
                    data.append(dict(sentence_id = sent_id, test_sentence = line))

        if globals.CONFIG["GENERAL"]["DEBUG"]:
            print(f"Loaded {len(data)} sentences from {globals.DATA_PATH}")
            for item in data:
                print(f"Sentence ID\t: {item["sentence_id"]}")
                print(f"Test Sentence\t: {item["test_sentence"]}")
                print()
            print("############################################")
        
        return data