import globals
from openai import OpenAI
import time

class APICaller:

    @staticmethod
    def read_instruction():
        # obtaining instruction file path from argument parameters
        path = "instructions/" + globals.EXPERIMENT_MODE + "/" + str(globals.PROMPT_ID) + ".txt"

        try:
            # reading instruction file
            with open(path, "r", encoding="utf-8") as file:
                instruction = file.read().strip()

            if not instruction:
                raise ValueError("Instruction file is empty. Please check the file content.")

            return instruction

        except FileNotFoundError:
            print(f"Error: File not found -> {path}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @staticmethod
    def run(data):
        # empty list to store api call results
        result = []

        """
        June 2025
            model snapshot: chatgpt-4o-latest (gpt-4o-2024-08-06)
            context window: 128,000 
            max output tokens: 16,384 
            knowledge cutoff: Oct 01, 2023 
            rate limits: 500 RPM (Tier 1)
        """
        if globals.MODEL == "gpt-4o":
            # not exceed the rate limit of 500 RPM (5 x 60 = 300 RPM)
            request_delay = globals.CONFIG["OPENAI_API"]["DELAY_PER_REQUEST"]

            # make sure to set your OPENAI_API_KEY in config.yaml file
            client = OpenAI(api_key = globals.CONFIG["OPENAI_API"]["OPENAI_API_KEY"])

            for item in data:
                print(f'Processing sentence {item["sentence_id"]}/{len(data)}')
                
                response = client.responses.create(
                    model = globals.MODEL,
                    temperature = globals.CONFIG["OPENAI_API"]["TEMPERATURE"],
                    max_output_tokens = globals.CONFIG["OPENAI_API"]["MAX_TOKENS"],
                    instructions = APICaller.read_instruction(),
                    input = "<input>" + item["test_sentence"] + "</input>"
                )
                
                result.append(dict(
                    sentence_id=item["sentence_id"],
                    test_sentence=item["test_sentence"],
                    corrected_sentence=response.output_text,
                    response=response.output
                ))

                time.sleep(request_delay)
            
            return result
        else:
            raise ValueError(f"Model {globals.MODEL} is not supported.")