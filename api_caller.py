import globals
import os
from datetime import datetime
from openai import OpenAI
from reporter import Reporter
import time
from google import genai
from google.genai import types

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

        if globals.MODEL in ["gpt-3.5-turbo", "gpt-4o", "gpt-4.1-2025-04-14", "gpt-5-mini-2025-08-07"]:
            # not exceed the rate limit of 500 RPM (5 x 60 = 300 RPM)
            request_delay = globals.CONFIG["OPENAI_API"]["DELAY_PER_REQUEST"]

            # make sure to set your OPENAI_API_KEY in config.yaml file
            client = OpenAI(api_key = globals.CONFIG["OPENAI_API"]["KEY"])

            # system instruction
            system_instruction = APICaller.read_instruction()

            for item in data:
                print(f'Processing sentence {item["sentence_id"]}/{len(data)}')
                
                # API call
                response = client.responses.create(
                    model = globals.MODEL,
                    temperature = globals.CONFIG["OPENAI_API"]["TEMP"],
                    instructions = system_instruction,
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
        
        elif globals.MODEL in ["gemini-3-flash-preview", "gemini-3-pro-preview"]:
            if globals.MODEL == "gemini-3-pro-preview":
                # not exceed the rate limit of 60-120 RPM (2 x 60 = 120 RPM)
                request_delay = globals.CONFIG["GEMINI_API"]["DELAY_PER_REQUEST_FOR_PRO"]
            else:
                # not exceed the rate limit of 500 RPM (5 x 60 = 300 RPM)
                request_delay = globals.CONFIG["GEMINI_API"]["DELAY_PER_REQUEST"]

            # make sure to set your GEMINI_API_KEY in config.yaml file
            client = genai.Client(api_key=globals.CONFIG["GEMINI_API"]["KEY"])

            # system instruction
            system_instruction = APICaller.read_instruction()

            for item in data:
                print(f'Processing sentence {item["sentence_id"]}/{len(data)}')

                # API call
                response = client.models.generate_content(
                        model = globals.MODEL,
                        contents = "<input>" + item["test_sentence"] + "</input>",
                        config = types.GenerateContentConfig(
                            system_instruction = system_instruction,
                            temperature = globals.CONFIG["GEMINI_API"]["TEMP"]
                        )
                    )
                
                # write results in a dictionary
                res_dict = dict(
                        sentence_id = item["sentence_id"],
                        test_sentence = item["test_sentence"],
                        corrected_sentence = response.text,
                        response = str(response) # Tüm objeyi loglamak için
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