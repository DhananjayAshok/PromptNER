import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, \
    GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM
import openai

import utils

openai.api_key = os.getenv("OPENAI_API_KEY")



class GPT3:
    model = "text-davinci-003"
    seconds_per_query = (60 / 20) + 0.01
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=GPT3.model, prompt=prompt, max_tokens=150)

    @staticmethod
    def decode_response(response):
        return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        return GPT3.decode_response(GPT3.request_model(prompt))


class HugginFaceModel:
    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(utils.Parameters.device)
        attention_mask = inputs.attention_mask.to(utils.Parameters.device)
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def __call__(self, prompt):
        return self.query(prompt)


class T5(HugginFaceModel):
    def __init__(self, size="large"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}").to(utils.Parameters.device)
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", model_max_length=600)


class GPTNeoX(HugginFaceModel):
    def __init__(self):
        self.model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").to(utils.Parameters.device)
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


class GPTNeo(HugginFaceModel):
    def __init__(self):
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(utils.Parameters.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


class GPTJ(HugginFaceModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(utils.Parameters.device)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def query(self, prompt):
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        outputs = self.model.generate(input_ids, attention_mask=attention_mask, do_sample=True, temperature=0.3,
                                      max_new_tokens=150)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

    def __call__(self, prompt):
        return self.query(prompt)
