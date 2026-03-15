from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class LLM:

    def _getResponse(self, prompt, context):  #  -> tuple[str, int, int]
        client = OpenAI(api_key=OPENAI_API_KEY)

        return client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
        )
