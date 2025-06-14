import globals
from openai import OpenAI

class APICaller:

    @staticmethod
    def read_instruction():
        # obtaining instruction file path from argument parameters
        path = "prompts/" + globals.EXPERIMENT_MODE + "/" + globals.PROMPT_ID + ".txt"

        try:
            # reading instruction file
            with open(path, "r", encoding="utf-8") as f:
                instruction = f.read().strip()

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
        # reading corresponding prompt
        instruction = APICaller.read_instruction()

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

            for idx, item in enumerate(data):
                print(f'Processing sentence {item.sent_id}/{len(data)}')
                
                """
                response = client.responses.create(
                    model = "gpt-4.1",
                    instructions = instruction
                    input = "How would I declare a variable for a last name?",
                )
                """


        else:
            raise ValueError(f"Model {globals.MODEL} is not supported.")