from algorithms import Algorithm, ConllConfig, GeniaConfig
from models import *

from data import *
from eval import f1


class Quick:
    @staticmethod
    def genia(i, model=GPT3.query, verbose=False):
        q = genia_train.loc[i]
        para = q['text']
        entities = q['entities']
        print(f"Paragraph: {para}\nEntities: {entities}")
        e = Algorithm(para=para)
        config = GeniaConfig()
        config.set_config(e)
        return entities, e.perform(model, verbose=verbose)[0]

    @staticmethod
    def conll(i, model=GPT3.query, verbose=False):
        q = conll_train.loc[i]
        para = q['text']
        entities = q['entities']
        e.set_para(para)
        e.set_model_fn(model)
        print(f"Paragraph: {para}\nEntities: {entities}")
        return entities, e.perform(verbose=verbose)[0]


e = Algorithm(split_phrases=True)
config = ConllConfig()
config.set_config(e, exemplar=True)
genia_train = load_genia()
conll_train = load_conll2003()
gc = GeniaConfig()
cc = ConllConfig()


if __name__ == "__main__":
    #loop(simple_q_a)
    pass
