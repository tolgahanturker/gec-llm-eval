import globals
import os
from datetime import datetime
from openai import OpenAI
from reporter import Reporter
import time

class APICaller:

    @staticmethod
    def read_instruction():
        try:
            # reading instruction file
            with open(globals.PROMPT_PATH, "r", encoding="utf-8") as file:
                instruction = file.read().strip()

            if not instruction:
                raise ValueError("Instruction file is empty. Please check the file content.")

            return instruction
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @staticmethod
    def run(data):
        # empty list to store api call results
        result = []

        # result file format: <model>_<input>_<prompt>_<zaman>.jsonl
        llm_output_filename = f"{globals.MODEL}_{os.path.basename(globals.DATA_PATH)}_{os.path.basename(globals.PROMPT_PATH)}_{datetime.now().strftime("%Y%m%d%H%M")}.jsonl"
        llm_output_path = os.path.join("results", llm_output_filename)

        if globals.MODEL == "gpt-5.1-2025-11-13":
            """
            -- Nov 2025 --
            model snapshot: GPT-5.1 (gpt-5.1-2025-11-13)
            context window: 400,000
            max output tokens: 128,000
            knowledge cutoff: Sep 30, 2024
            rate limits: 500 RPM (Tier 1)
            """
            # not exceed the rate limit of 500 RPM (5 x 60 = 300 RPM)
            request_delay = globals.CONFIG["OPENAI_API"]["DELAY_PER_REQUEST"]

            # make sure to set your OPENAI_API_KEY in config.yaml file
            client = OpenAI(api_key = globals.CONFIG["OPENAI_API"]["KEY"])

            for item in data:
                print(f'Processing sentence {item["sentence_id"]}/{len(data)}')
                
                # API call
                response = client.responses.create(
                    model = globals.MODEL,
                    temperature = globals.CONFIG["OPENAI_API"]["TEMP"],
                    instructions = APICaller.read_instruction(),
                    input = "<input>" + item["test_sentence"] + "</input>"
                )
                
                # write results in a dictionary
                res_dict = dict(
                    sentence_id = item["sentence_id"],
                    test_sentence = item["test_sentence"],
                    corrected_sentence = response.output_text,
                    response = str(response.output)
                )

                # write result in memory
                result.append(res_dict)

                # store result incrementally to avoid data loss
                Reporter.write_jsonl(llm_output_path, res_dict)

                # delay to avoid exceeding rate limits
                time.sleep(request_delay)
            
            return result, llm_output_path
        else:
            raise ValueError(f"Model {globals.MODEL} is not supported.")