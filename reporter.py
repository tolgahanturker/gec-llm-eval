import json
import globals
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize

class Reporter:

    @staticmethod
    def write_jsonl(path, data):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def write_results_for_eval(results, path_to_read):
    
        if "conll2014" in globals.DATA_PATH:
            result_filename = f"{globals.MODEL}_{os.path.basename(globals.DATA_PATH)}_{os.path.basename(globals.PROMPT_PATH)}_{datetime.now().strftime("%Y%m%d%H%M")}.txt"
            result_path = os.path.join("results", result_filename)

            with open(path_to_read, 'r', encoding='utf-8') as infile, open(result_path, "w", encoding="utf-8") as outfile:
                for line in infile:
                    json_line = json.loads(line)
                    
                    # NLTK word tokenization for compatibility with CoNLL-2014 shared task results
                    tokens = word_tokenize(json_line['corrected_sentence'], language='english')
                    tokenized_sentence = " ".join(tokens)
                    
                    # the file used for evaluation with m2scorer
                    outfile.write(tokenized_sentence + '\n')
        else:
            return