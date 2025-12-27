from pathlib import Path
import random

FILE_PATH = "data/wandilocness/m2/ABCN.dev.gold.bea19.m2" # BEA-2019 Dev set path
OUTPUT_PATH = "data/sampled/sampled_wandilocness_dev.m2"
SAMPLE_SIZE = 354 # represntative sample size according to Cochran's formula
SEED = 17 # for reproducibility

def main():
    # reading BEA-2019 Dev m2 file
    with Path(FILE_PATH).open(encoding="utf-8") as file:
        # reading file 
        content = file.read().strip()
        # splitting into sentences based on double new lines
        sentences = content.split('\n\n')
        # removing any empty blocks
        sentences = [s for s in sentences if s.strip()]

        print(f"Total sentences: {len(sentences)}")

        # selecting random sample of sentences
        random.seed(SEED)
        selected_indices = random.sample(range(len(sentences)), SAMPLE_SIZE)
        selected_indices.sort()

        print(f"Selected indices: {selected_indices}")

        # extracting selected sentences
        selected_blocks = [sentences[i] for i in selected_indices]

        # writing selected sentences to new m2 file
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(selected_blocks) + '\n\n') 


if __name__ == "__main__":
    main()