from pathlib import Path
import globals

class DataLoader:
    @staticmethod
    def load_data():
        sent_id = 0
        result = []

        with Path(globals.INPUT_FILE_PATH).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                sent_id += 1
                result.append(dict(sentence_id = sent_id, test_sentence = line))
                
        return result