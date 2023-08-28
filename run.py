import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm
import time
import pandas as pd
import openai
from seqeval.metrics import f1_score


def eval_dataset(val, model, algorithm, sleep_between_queries=None, print_every=10):
    algorithm.set_model_fn(model)
    columns = ["text", "entities", "truth", "pred", "meta", "f1"]
    data = []
    preds, truths = [], []
    for i, info in tqdm(enumerate(val.iterrows()), total=len(val)):
        index, q = info
        para = q['text']
        entities = q['entities']
        subdata = [para, entities, q['exact_types']]
        algorithm.set_para(para)
        if sleep_between_queries is not None:
            time.sleep(sleep_between_queries)
        types = None
        flag = False
        while not flag:
            try:
                true_tokens = None
                if "true_tokens" in val.columns:
                    true_tokens = q["true_tokens"]
                span_pred, meta = algorithm.perform_span(true_tokens=true_tokens, verbose=False)
                p = [span_pred]
                t = [q['exact_types']]
                preds.append(span_pred)
                truths.append(q['exact_types'])
                mini_f1 = f1_score(t, p)
                subdata.extend([span_pred, meta, mini_f1])
                data.append(subdata)
                f1_micro = f1_score(truths, preds, average="micro")
                flag = True
            except openai.error.RateLimitError:
                time.sleep(0.5)
            except IndexError:
                flag = True
        if print_every is not None:
            if i % print_every == 0:
                f1_micro = f1_score(truths, preds, average="micro")
                f1_macro = f1_score(truths, preds, average="macro")
                print(f"Iteration {i}: micro f1: {f1_micro}, macro f1: {f1_macro}")
    f1_micro = f1_score(truths, preds, average="micro")
    f1_macro = f1_score(truths, preds, average="macro")
    print(f"Finally: micro f1: {f1_micro}, macro f1: {f1_macro}")
    df = pd.DataFrame(data=data, columns=columns)
    return f1_micro, f1_macro, df


def complete_eval(dataset, model, algorithm, n_runs=2, sleep_between_queries=None, limit=None):
    micros = []
    macros = []
    for i in range(n_runs):
        if limit is not None:
            small_dataset = dataset.sample(limit)
        else:
            small_dataset = dataset
        f1_micro, f1_macro, df = eval_dataset(small_dataset, model, algorithm, sleep_between_queries=sleep_between_queries)
        micros.append(f1_micro)
        macros.append(f1_macro)
    micros = np.array(micros)
    macros = np.array(macros)
    return micros, macros, df


def eval_conll(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    config = ConllConfig()
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        conll_train = load_conll2003("train")
        subsample = sample_all_types(conll_train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    conll = load_conll2003("test")
    return complete_eval(conll, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_genia(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    config = GeniaConfig()
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        train = load_genia()
        subsample = sample_all_types(train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    genia = load_genia()
    return complete_eval(genia, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_tweetner(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    config = TweetNERConfig()
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        train = load_tweetner("train")
        subsample = sample_all_types(train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["true_tokens"].tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    tweetner = load_tweetner("validation")
    return complete_eval(tweetner, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_fabner(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    config = FabNERConfig()
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        train = load_fabner("train")
        subsample = sample_all_types(train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    fabner = load_fabner("test")
    return complete_eval(fabner, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_cross_ner(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    cats = ['politics', 'literature', 'ai', 'science', 'music']
    confs = [CrossNERPoliticsConfig(), CrossNERLiteratureConfig(), CrossNERAIConfig(),
             CrossNERNaturalSciencesConfig(), CrossNERMusicConfig()]
    category = kwargs.get('add_info', None)
    assert category in cats
    i = cats.index(category)
    config = confs[i]
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        train = load_cross_ner(category=category, split="train")
        subsample = sample_all_types(train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    dataset = load_cross_ner(category=category, split="test")
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_few_nerd_intra(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, autogen=True, **kwargs):
    splits = ["test"]
    confs = [FewNERDINTRATestConfig()]
    split = kwargs.get("add_info")
    assert split in splits
    i = splits.index(split)
    config = confs[i]
    algorithm.split_phrases = False
    if autogen:
        algorithm.set_model_fn(model)
        train = load_few_nerd(category="intra", split=split)
        subsample = sample_all_types(train, 3)
        texts = subsample["text"].tolist()
        tokens = subsample["text"].apply(lambda x: x.split(" ")).tolist()
        labels = subsample["exact_types"].tolist()
        config.autogenerate_annotations(algorithm, texts, tokens, labels)
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    dataset = load_few_nerd(category="intra", split=split)
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def run(dataset="conll", subdataset=None, gpt=True, exemplar=True, coT=True, defn=True, tf=True, name_meta=""):
    print(f"Running for: {dataset}, {subdataset}")
    res_path = "results"
    gpt_limit = 20
    gpt_nruns = 1
    other_limit = 200
    other_nruns = 2
    Algorithm_class = Algorithm

    if dataset == "conll":
        eval_fn = eval_conll
    elif dataset == "genia":
        eval_fn = eval_genia
    elif dataset == "crossner":
        eval_fn = eval_cross_ner
    elif dataset == "fewnerd":
        eval_fn = eval_few_nerd_intra
    elif dataset == "tweetner":
        eval_fn = eval_tweetner
    elif dataset == "fabner":
        eval_fn = eval_fabner
    else:
        raise ValueError(f"Unknown Dataset: {dataset}")

    if gpt:
        model = OpenAIGPT()
        micros, macros, df = eval_fn(model, Algorithm_class(), n_runs=gpt_nruns,
                                                      sleep_between_queries=model.seconds_per_query,
                                                      limit=gpt_limit,
                                                      exemplar=exemplar, coT=coT, defn=defn, tf=tf,
                                                      add_info=subdataset)
    else:
        model = Alpaca(size='base')
        micros, macros, df = eval_fn(model, Algorithm_class(), n_runs=other_nruns,
                                                      sleep_between_queries=None, exemplar=exemplar,
                                                      coT=coT, defn=defn, tf=tf,
                                                      limit=other_limit, add_info=subdataset)
    print(f"Final Results For {name_meta} | {dataset} {'('+subdataset+')' if subdataset is not None else ''}) "
          f"|CoT {coT} | Exemplar {exemplar} (tf {tf}) |Defn {defn}")
    print(f"Micro f1_means: {micros.mean()}")
    print(f"Micro f1_stds: {micros.std()}")
    print(f"Macro f1_means: {macros.mean()}")
    print(f"Macro f1_stds: {macros.std()}")
    save_path = f"results/{name_meta}{dataset}{subdataset}.csv"
    df.to_csv(save_path, index=False)
    return micros, macros


def run_all_datasets(gpt=False, exemplar=True, coT=True, defn=True, tf=True,
                     name_meta="",
                     dataset_exclude=[], subdataset_exclude=[]):
    d = {}
    datasets = ["conll", "genia", "crossner", "fewnerd", "tweetner", "fabner"]
    subdatasets = {"crossner": ['politics', 'literature', 'ai', 'science', 'music'],
                   'fewnerd': ["test"]}
    for dataset in datasets:
        if dataset in dataset_exclude:
            continue
        sub = subdatasets.get(dataset, None)
        if sub is None:
            micro, macro = run(gpt=gpt, dataset=dataset, coT=coT, exemplar=exemplar, defn=defn, tf=tf,
                               name_meta=name_meta)
            d[dataset] = [(macro * 100).mean(), (macro * 100).std(), (micro * 100).mean(), (micro * 100).std()]
        else:
            for s in sub:
                if s in subdataset_exclude:
                    continue
                macro, micro = run(gpt=gpt, dataset=dataset, subdataset=s,
                                   coT=coT, exemplar=exemplar, defn=defn, tf=tf, name_meta=name_meta)
                d[f"{dataset}_{s}"] = [(macro * 100).mean(), (macro * 100).std(),
                                       (micro * 100).mean(), (micro * 100).std()]
    return d


def ablate_all(gpt=False, vary_cot=True, vary_exemplar=True, vary_tf=True, vary_defn=True,
               dataset_exclude=["genia"], subdataset_exclude=[]):
    cot_options = [True, False] if vary_cot else [True]
    exemplar_options = [True, False] if vary_exemplar else [True]
    tf_options = [True, False] if vary_tf else [True]
    defn_options = [True, False] if vary_defn else [True]
    # first take off cot then tf then example then defn
    res_d = {}
    for defn in defn_options:
        for exemplar in exemplar_options:
            for cot in cot_options:
                for tf in tf_options:
                    key = (defn, exemplar, cot, tf)
                    res_d[key] = run_all_datasets(gpt=gpt, exemplar=exemplar, coT=cot, defn=defn, tf=tf,
                     dataset_exclude=dataset_exclude, subdataset_exclude=subdataset_exclude)

    print(f"Ablations Done.... \nFinal Results For All: f1 Macro Mean, f1 Macro Std, f1 Micro Mean, f1 Micro Std")
    for defn in defn_options:
        for exemplar in exemplar_options:
            for cot in cot_options:
                for tf in tf_options:
                    key = (defn, exemplar, cot, tf)
                    print(f"Defn: {key[0]}\tExemplar: {key[1]}\tCoT: {key[2]}\ttf:{key[3]}")
                    for dataset_key in res_d[key]:
                        print(f"\t{dataset_key}")
                        formatted = [f"{i:.3f}" for i in res_d[key][dataset_key]]
                        print(f"\t\t{formatted}")
    return


def ablate_best(gpt=False, dataset_exclude=["genia"], subdataset_exclude=["politics", "literature", "train", "dev"]):
    configurations = [(True, True, True, True), (False, True, True, True),
                      (True, False, True, True), (True, True, False, True), (True, True, True, False)]
    res_d = {}
    for defn, exemplar, cot, tf in configurations:
        key = (defn, exemplar, cot, tf)
        res_d[key] = run_all_datasets(gpt=gpt, exemplar=exemplar, coT=cot, defn=defn, tf=tf,
         dataset_exclude=dataset_exclude, subdataset_exclude=subdataset_exclude)

    print(f"Ablations Done.... \nFinal Results For All: f1 Macro Mean, f1 Macro Std, f1 Micro Mean, f1 Micro Std")
    for defn, exemplar, cot, tf in configurations:
        key = (defn, exemplar, cot, tf)
        print(f"Defn: {key[0]}\tExemplar: {key[1]}\tCoT: {key[2]}\ttf:{key[3]}")
        for dataset_key in res_d[key]:
            print(f"\t{dataset_key}")
            formatted = [f"{i:.3f}" for i in res_d[key][dataset_key]]
            print(f"\t\t{formatted}")
    return


if __name__ == "__main__":
    from models import OpenAIGPT, T5XL, Alpaca
    run(gpt=True)

