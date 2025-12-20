import json
import globals
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import errant
import spacy

class Reporter:

    @staticmethod
    def write_jsonl(path, data):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    @staticmethod
    def write_results_for_eval_m2(results, path_to_read):
        result_filename = f"{globals.MODEL}_{os.path.basename(globals.DATA_PATH)}_{os.path.basename(globals.PROMPT_PATH)}_{datetime.now().strftime("%Y%m%d%H%M")}.m2"
        result_path = os.path.join("results", result_filename)
        
        # for ERRANT compatible tokenization of hypothesis result
        nlp = spacy.load('en_core_web_sm')
        annotator = errant.load('en', nlp)
        with open(path_to_read, 'r', encoding='utf-8') as infile, open(result_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                json_line = json.loads(line)
                
                test_sentence_doc = nlp(json_line['test_sentence'])
                corrected_sentence_doc = nlp(json_line["corrected_sentence"])
                
                source_tokens = [token.text for token in test_sentence_doc]
                outfile.write("S " + " ".join(source_tokens) + "\n")

                edits = annotator.annotate(test_sentence_doc, corrected_sentence_doc)
                for edit in edits:
                    edit_output = edit.to_m2(id=0)
                    outfile.write(edit_output + "\n")
                outfile.write("\n")

    @staticmethod
    def write_results_for_eval(results, path_to_read):
        
        # in any case, write m2 file for ERRANT evaluation
        Reporter.write_results_for_eval_m2(results, path_to_read)
            
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
        