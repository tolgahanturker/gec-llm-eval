from sacremoses import MosesDetokenizer

class Tokenizer:

    # for detokenization of input sentences
    md = MosesDetokenizer(lang='en')

    @staticmethod
    def davis_detokenize(token_list):
        # Detokenization strategy used in the paper (doi: 10.18653/v1/2024.findings-acl.711) by Christopher Davis et al.
        # Steps:
        # 1. Applying standart Moses detokenization.
        # 2. Combining negative contractions missed by Moses.

        text = Tokenizer.md.detokenize(token_list)
        text = text.replace(" n't", "n't")

        return text.strip()