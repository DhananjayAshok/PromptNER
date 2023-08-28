import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai

import utils

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGPT:
    #model = "text-davinci-003"
    #model = "gpt-4"
    model = "gpt-3.5-turbo"
    #model = "davinci"
    seconds_per_query = (60 / 20) + 0.01
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=OpenAIGPT.model, prompt=prompt, max_tokens=250)

    @staticmethod
    def request_chat_model(msgs):
        messages = []
        for message in msgs:
            content, role = message
            messages.append({"role": role, "content": content})
        return openai.ChatCompletion.create(model=OpenAIGPT.model, messages=messages)

    @staticmethod
    def decode_response(response):
        if OpenAIGPT.is_chat():
            return response["choices"][0]["message"]["content"]
        else:
            return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        return OpenAIGPT.decode_response(OpenAIGPT.request_model(prompt))

    @staticmethod
    def chat_query(msgs):
        return OpenAIGPT.decode_response(OpenAIGPT.request_chat_model(msgs))

    @staticmethod
    def is_chat():
        return OpenAIGPT.model in ["gpt-4", "gpt-3.5-turbo"]

    @staticmethod
    def __call__(inputs):
        if OpenAIGPT.is_chat():
            return OpenAIGPT.chat_query(inputs)
        else:
            return OpenAIGPT.query(inputs)


class HugginFaceModel:
    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(utils.Parameters.devices[0])
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def __call__(self, prompt):
        return self.query(prompt)


class T5(HugginFaceModel):
    def __init__(self, size="large"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}").to(utils.Parameters.devices[0])
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", model_max_length=600)


class ParallelHuggingFaceModel(HugginFaceModel):
    def parallel(self, num_layers=24, num_devices=4):
        self.devices = utils.Parameters.get_device_ints(num_devices)
        layer_per_device = num_layers // num_devices
        device_map = {}
        start = 0
        for i, device in enumerate(self.devices):
            if i == len(self.devices) - 1:
                device_map[device] = [j for j in range(start, num_layers)]
            else:
                device_map[device] = [j for j in range(start, start+layer_per_device)]
                start = start + layer_per_device
        self.model.parallelize(device_map)

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.devices[0])
        outputs = self.model.generate(**inputs, max_new_tokens=600)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


class T5XL(ParallelHuggingFaceModel):
    def __init__(self, size="xxl"):
        assert size in ["xl", "xxl"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", model_max_length=600)
        self.parallel(num_layers=24, num_devices=4)


class Alpaca(ParallelHuggingFaceModel):
    def __init__(self, size="base"):
        assert size in ["base", "large",  "gpt4-xl", "xl", "xxl"]
        layer_sizes = {"base": 12, "large": 24, "xl": 24, "gpt4-xl": 24, "xxl": 24}
        self.tokenizer = AutoTokenizer.from_pretrained(f"declare-lab/flan-alpaca-{size}", model_max_length=600)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"declare-lab/flan-alpaca-{size}")
        self.parallel(num_layers=layer_sizes[size], num_devices=4)