import os
import openai
from algorithms import Algorithm, ConllConfig, GeniaConfig

from data import *

openai.api_key = os.getenv("OPENAI_API_KEY")
model = "text-davinci-003"


def request_model(prompt):
    return openai.Completion.create(model=model, prompt=prompt, max_tokens=200)


def decode_response(response):
    return response["choices"][0]["text"]


def query(prompt):
    return decode_response(request_model(prompt))


train = get_re_dset(i=2)
para = train.token[1]
e = Algorithm(para=para)

genia_train = load_genia()
conll_train = load_conll2003()
gc = GeniaConfig()
cc = ConllConfig()


def quick(i):
    para = train.token[i]
    print(f"Paragraph: {para}")
    e = Algorithm(para=para)
    return e.perform(query)


def genia(i, verbose=False):
    q = genia_train.loc[i]
    para = q['text']
    entities = q['entities']
    print(f"Paragraph: {para}\nEntities: {entities}")
    e = Algorithm(para=para)
    config = GeniaConfig()
    config.set_config(e)
    return e.perform(query, verbose=verbose)


def conll(i, verbose=False):
    q = conll_train.loc[i]
    para = q['text']
    entities = q['entities']
    print(f"Paragraph: {para}\nEntities: {entities}")
    e = Algorithm(para=para)
    config = ConllConfig()
    config.set_config(e, exemplar=True)
    return e.perform(query, verbose=verbose)


if __name__ == "__main__":
    #loop(simple_q_a)
    pass
