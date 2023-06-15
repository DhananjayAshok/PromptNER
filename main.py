from algorithms import *
from models import *

from data import *
from eval import f1, is_eq, type_f1
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
        ret = e.perform(verbose=verbose)
        ans = ret[0]
        Quick.analyze(q, ret)
        return

    @staticmethod
    def analyze(q, ret):
        ans = ret[0]
        entities = q['entities']
        true_types = q['types']
        print(true_types)
        print(ans)
        entities_small = [e.lower().strip().strip(string.punctuation).strip() for e in entities]
        print(f"False positives: {set(entities_small).difference(set(ans))}")
        print(f"False negatives: {set(ans).difference(set(entities_small))}")
        print(f"F1: {f1(entities, ans)}")
        if e.identify_types:
            types = ret[1]
            print("Type f1: ")
            print(type_f1(q, ans, types))


    @staticmethod
    def genia(i, model=OpenAIGPT(), verbose=True):
        config = GeniaConfig()
        return Quick.dataset(i, train_dset=genia_train, config=config, verbose=verbose)

    @staticmethod
    def conll(i, model=OpenAIGPT(), verbose=True):
        config = ConllConfig()
        return Quick.dataset(i, train_dset=conll_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def crossner(i, model=OpenAIGPT(), verbose=True, category="politics"):
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
        confs = [FewNERDINTRATrainConfig(), FewNERDINTRADevConfig(), FewNERDINTRATestConfig()]
        assert split in splits
        j = splits.index(split)
        config = confs[j]
        return Quick.dataset(i, train_dset=few_nerd_train, config=config, model=model, verbose=verbose)


e = Algorithm(split_phrases=True, identify_types=True)
genia_train = load_genia()
conll_train = load_conll2003()
few_nerd_train = load_few_nerd(split="train")
cross_ner_train = load_cross_ner(category="ai")


if __name__ == "__main__":
    #loop(simple_q_a)
    pass
