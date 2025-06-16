from pathlib import Path
import globals

class DataLoader:
    @staticmethod
    def load_data():
        sent_id = 0
        result = []

        if globals.INPUT_FILE_PATH.endswith(".m2"):
            with Path(globals.INPUT_FILE_PATH).open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if line.startswith("S "):
                        sent_id += 1
                        result.append(dict(sentence_id = sent_id, test_sentence = line[2:]))
        else:
            with Path(globals.INPUT_FILE_PATH).open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    sent_id += 1
                    result.append(dict(sentence_id = sent_id, test_sentence = line))
                
        
        if globals.CONFIG["GENERAL"]["DEBUG"]:
            print("####################")
            print(f"Loaded {len(result)} sentences from {globals.INPUT_FILE_PATH}")


        return result