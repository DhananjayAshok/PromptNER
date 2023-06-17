from algorithms import *
from models import *

from data import *
from seqeval.metrics import f1_score
import string


class Quick:
    @staticmethod
    def example_span(para, config=ConllConfig(), model = OpenAIGPT(), verbose=True):
        e.set_para(para)
        e.set_model_fn(model)
        e.split_phrases = False
        config.set_config(e, exemplar=True, coT=True, tf=True)
        ret = e.perform_span(verbose=verbose)
        return ret

    @staticmethod
    def dataset(i, train_dset, config, model=OpenAIGPT(), verbose=True):
        q = train_dset.loc[i]
        para = q['text']
        entities = q['entities']
        print(f"Paragraph: {para}")
        e.set_para(para)
        e.set_model_fn(model)
        config.set_config(e, exemplar=True, coT=True, tf=True)
        ret = e.perform_span(verbose=verbose)
        Quick.analyze(q, ret)
        return

    @staticmethod
    def analyze(q, ret):
        ans = [ret]
        entities = [q['exact_types']]
        true_types = q['types']
        print("True Types were: ", true_types)
        print(f"F1: {f1_score(entities, ans)}")


    @staticmethod
    def genia(i, model=OpenAIGPT(), verbose=True):
        config = GeniaConfig()
        return Quick.dataset(i, train_dset=genia_train, config=config, verbose=verbose)

    @staticmethod
    def conll(i, model=OpenAIGPT(), verbose=True):
        config = ConllConfig()
        return Quick.dataset(i, train_dset=conll_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def crossner(i, model=OpenAIGPT(), verbose=True, category="ai"):
        cats = ['politics', 'literature', 'ai', 'science', 'music']
        confs = [CrossNERPoliticsConfig(), CrossNERLiteratureConfig(), CrossNERAIConfig(),
                 CrossNERNaturalSciencesConfig(), CrossNERMusicConfig()]
        assert category in cats
        j = cats.index(category)
        print(j)
        config = confs[j]
        return Quick.dataset(i, train_dset=cross_ner_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def fewnerd(i, model=OpenAIGPT(), verbose=True, split="train"):
        splits = ["train", "dev", "test"]
        confs = [FewNERDINTRATestConfig()]
        assert split in splits
        j = splits.index(split)
        config = confs[j]
        return Quick.dataset(i, train_dset=few_nerd_train, config=config, model=model, verbose=verbose)


e = Algorithm(split_phrases=False, identify_types=True)
genia_train = load_genia()
#conll_train = load_conll2003()
few_nerd_train = load_few_nerd()
cross_ner_train = load_cross_ner(category="ai")


if __name__ == "__main__":
    #loop(simple_q_a)
    pass
