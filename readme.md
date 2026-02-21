# Instruction-Conditioned LLMs for Grammatical Error Correction


This repository contains the official implementation and evaluation framework for the paper: *"Instruction-Conditioned LLMs for Grammatical Error Correction: Analyzing Neutral-Anchored Instructional Sensitivity Across Editing Modes"*.

If you use this framework or our findings in your research, please cite our paper:

> Türker, T., Eryiğit, G. (2026). *Instruction-Conditioned LLMs for Grammatical Error Correction: Analyzing Neutral-Anchored Instructional Sensitivity Across Editing Modes*. [Conference Name TBD].


## Usage
Follow these steps to set up the environment and reproduce the experiments.

### Environment Setup
It is highly recommended to use a virtual environment (Python 3.9+) to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/tolgahanturker/gec-llm-eval.git
cd gec-llm-eval

# Create a virtual environment
python -m venv venv

# Activate the environment
# For Windows:
.\venv\Scripts\activate
# For Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# Download necessary spaCy model for ERRANT
python -m spacy download en_core_web_sm
```

### Running Experiments
Before running the experiments, you must configure your API keys in the `configs/config.yaml` file.

```bash
# Run inference with a specific model and instruction on desired benchmark
# python main.py <model> <data> <prompt>
python main.py gpt-5.1-2025-11-13 "data/test/conll2014/official-2014.combined-withalt.m2" "instructions/zero-shot_neutral.txt"
# python main.py gpt-5.1-2025-11-13 "data/test/jfleg/test.src" "instructions/zero-shot_neutral.txt"
# python main.py gpt-5.1-2025-11-13 "data/test/wandilocness/ABCN.dev.gold.bea19.m2" "instructions/zero-shot_neutral.txt"
```

### Evaluation of Model Output
```bash
# Run the evaluation script to calculate related scores
# python eval.py <metric> <system_output> <gold_data>
python eval.py m2scorer "results/gpt-5.1-2025-11-13_official-2014.combined-withalt.m2_zero-shot_neutral.txt_202512200016.txt" "data/test/conll2014/official-2014.combined-withalt.m2"
# python eval.py gleu "results/gpt-5.1-2025-11-13_test.src_zero-shot_neutral.txt_202512202227.txt" "./data/test/jfleg/test"
# python eval.py errant "results/gpt-5.1-2025-11-13_ABCN.dev.gold.bea19.m2_zero-shot_neutral.txt_202512210123.m2" "./data/test/wandilocness/ABCN.dev.gold.bea19.m2"
```