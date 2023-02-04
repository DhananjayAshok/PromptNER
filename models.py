import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3:
    model = "text-davinci-003"
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=GPT3.model, prompt=prompt, max_tokens=200)

    @staticmethod
    def decode_response(response):
        return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        return GPT3.decode_response(GPT3.request_model(prompt))


class T5:
    def __init__(self, size="large"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}")

    def query(self, prompt):
        inputs = self.tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

