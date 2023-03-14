from algorithms import *
from models import *

from data import *
from eval import f1


class Quick:
    @staticmethod
    def dataset(i, train_dset, config, model, verbose):
        q = train_dset.loc[i]
        para = q['text']
        entities = q['entities']
        print(f"Paragraph: {para}")
        e.set_para(para)
        e.set_model_fn(model)
        config.set_config(e, exemplar=True, coT=True)
        ans = e.perform(verbose=verbose)[0]
        Quick.analyze(entities, ans)
        return

    @staticmethod
    def analyze(entities, ans):
        print(entities)
        print(ans)
        entities_small = [e.lower() for e in entities]
        print(f"False positives: {set(entities_small).difference(set(ans))}")
        print(f"False negatives: {set(ans).difference(set(entities_small))}")
        print(f"F1: {f1(entities, ans)}")


    @staticmethod
    def genia(i, model=GPT3.query, verbose=False):
        config = GeniaConfig()
        config.set_config(e)
        return Quick.dataset(i, train_dset=genia_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def conll(i, model=GPT3.query, verbose=False):
        config = ConllConfig()
        config.set_config(e)
        return Quick.dataset(i, train_dset=conll_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def crossner(i, model=GPT3.query, verbose=False, category="politics"):
        cats = ['politics', 'literature', 'ai', 'science', 'music']
        confs = [CrossNERPoliticsConfig(), CrossNERLiteratureConfig(), CrossNERAIConfig(),
                 CrossNERNaturalSciencesConfig(), CrossNERMusicConfig()]
        assert category in cats
        j = cats.index(category)
        print(j)
        config = confs[j]
        config.set_config(e)
        return Quick.dataset(i, train_dset=cross_ner_train, config=config, model=model, verbose=verbose)

    @staticmethod
    def fewnerd(i, model=GPT3.query, verbose=False, split="train"):
        splits = ["train", "dev", "test"]
        confs = [FewNERDINTRATrainConfig(), FewNERDINTRADevConfig(), FewNERDINTRATestConfig()]
        assert split in splits
        j = splits.index(split)
        config = confs[j]
        config.set_config(e)
        return Quick.dataset(i, train_dset=few_nerd_train, config=config, model=model, verbose=verbose)


e = Algorithm(split_phrases=True)
genia_train = load_genia()
conll_train = load_conll2003()
few_nerd_train = load_few_nerd(split="train")
cross_ner_train = load_cross_ner(category="ai")


if __name__ == "__main__":
    #loop(simple_q_a)
    pass
