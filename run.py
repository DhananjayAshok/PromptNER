import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm
import time
import pandas as pd
import openai
from eval import f1, is_eq


def eval_dataset(val, model, algorithm, sleep_between_queries=None, print_every=10):
    algorithm.set_model_fn(model)
    f1s = []
    tp, fp, fn = 0, 0, 0
    mistake_data = []
    for i, info in tqdm(enumerate(val.iterrows()), total=len(val)):
        index, q = info
        para = q['text']
        entities = q['entities']
        algorithm.set_para(para)
        if sleep_between_queries is not None:
            time.sleep(sleep_between_queries)
        flag = False
        while not flag:
            try:
                print(f"Paragraph: {algorithm.para}")
                preds, metadata = algorithm.perform(verbose=True)
                flag = True
            except openai.error.RateLimitError:
                time.sleep(0.5)
        f1_score, tp_a, fp_a, fn_a = f1(entities, preds)
        tp += tp_a
        fp += fp_a
        fn += fn_a
        if f1_score != 1:
            mistake_data.append([index, para, entities, preds, metadata, f1_score])
        f1s.append(f1_score)
        if print_every is not None:
            if i % print_every == 0:
                avg_f1 = np.array(f1s).mean()
                std_f1 = np.array(f1s).std()
                if (tp + 0.5 * (fp + fn)) == 0:
                    micro_f1 = 0
                else:
                    micro_f1 = tp / (tp + 0.5 * (fp + fn))
                print(f"Iteration {i}: Avg f1: {avg_f1}, Std f1: {std_f1}, micro f1: {micro_f1}")
    f1s = np.array(f1s)
    if (tp + 0.5 * (fp + fn)) == 0:
        micro_f1 = 0
    else:
        micro_f1 = tp / (tp + 0.5 * (fp + fn))
    return f1s.mean(), f1s.std(), micro_f1, mistake_data


def complete_eval(dataset, model, algorithm, n_runs=2, sleep_between_queries=None, limit=None):
    f1_means, f1_stds, micro_f1s = [], [], []
    mistake_columns = ["idx", "para", "entities", "preds", "meta", "f1"]
    for i in range(n_runs):
        if limit is not None:
            small_dataset = dataset.sample(limit)
        else:
            small_dataset = dataset
        f1_mean, f1_std, micro_f1, mistake_data = eval_dataset(small_dataset, model, algorithm, sleep_between_queries=sleep_between_queries)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)
        micro_f1s.append(micro_f1)
    df = pd.DataFrame(data=mistake_data, columns=mistake_columns)
    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_std)
    micro_f1s = np.array(micro_f1s)
    return f1_means, f1_stds, micro_f1s, df


def eval_conll(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, **kwargs):
    config = ConllConfig()
    algorithm.split_phrases = True
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    conll = load_conll2003("validation")
    return complete_eval(conll, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_genia(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, **kwargs):
    config = GeniaConfig()
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    genia = load_genia()
    return complete_eval(genia, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_cross_ner(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, **kwargs):
    cats = ['politics', 'literature', 'ai', 'science', 'music']
    confs = [CrossNERPoliticsConfig(), CrossNERLiteratureConfig(), CrossNERAIConfig(),
             CrossNERNaturalSciencesConfig(), CrossNERMusicConfig()]
    category = kwargs.get('add_info', None)
    assert category in cats
    i = cats.index(category)
    config = confs[i]
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    dataset = load_cross_ner(category=category)
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_few_nerd_intra(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True,  **kwargs):
    splits = ["train", "dev", "test"]
    confs = [FewNERDINTRATrainConfig(), FewNERDINTRADevConfig(), FewNERDINTRATestConfig()]
    split = kwargs.get("add_info")
    assert split in splits
    i = splits.index(split)
    config = confs[i]
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf)
    dataset = load_few_nerd(category="intra", split=split)
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def run(dataset="conll", subdataset=None, gpt=False, exemplar=True, coT=True, defn=True, tf=True, name_meta=""):
    res_path = "results"
    gpt_limit = 20
    gpt_nruns = 2
    other_limit = 100
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

    if gpt:
        model = OpenAIGPT()
        f1_mean, f1_std, micro_f1, mistakes = eval_fn(model, Algorithm_class(), n_runs=gpt_nruns,
                                                      sleep_between_queries=model.seconds_per_query,
                                                      limit=gpt_limit,
                                                      exemplar=exemplar, coT=coT, defn=defn, tf=tf,
                                                      add_info=subdataset)
    else:
        model = T5XL(size='xxl')
        f1_mean, f1_std, micro_f1, mistakes = eval_fn(model, Algorithm_class(), n_runs=other_nruns,
                                                      sleep_between_queries=None, exemplar=exemplar,
                                                      coT=coT, defn=defn, tf=tf,
                                                      limit=other_limit, add_info=subdataset)
    print(f"Final Results For {name_meta} | {dataset} {'('+subdataset+')' if subdataset is not None else ''}) "
          f"|CoT {coT} | Exemplar {exemplar} (tf {tf}) |Defn {defn}")
    print(f"f1_means: {f1_mean}")
    print(f"f1_stds: {f1_std}")
    print(f"micro_f1s: {micro_f1}")
    print(f"Saving file to {res_path}/{name_meta}{model.__class__.__name__}_{dataset}{subdataset}.csv")
    mistakes.to_csv(f"{res_path}/{name_meta}{model.__class__.__name__}_{dataset}{subdataset}.csv")
    return f1_mean, micro_f1


def run_all_datasets(gpt=False, exemplar=True, coT=True, defn=True, tf=True,
                     name_meta="",
                     dataset_exclude=["genia"], subdataset_exclude=[]):
    d = {}
    datasets = ["conll", "genia", "crossner", "fewnerd"]
    subdatasets = {"crossner": ['politics', 'literature', 'ai', 'science', 'music'],
                   'fewnerd': ["train", "dev", "test"]}
    for dataset in datasets:
        if dataset in dataset_exclude:
            continue
        sub = subdatasets.get(dataset, None)
        if sub is None:
            macro, micro = run(gpt=gpt, dataset=dataset, coT=coT, exemplar=exemplar, defn=defn, tf=tf,
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


if __name__ == "__main__":
    from models import OpenAIGPT, T5XL
    run_all_datasets(gpt=True, dataset_exclude=["conll", "genia"], subdataset_exclude=["politics", "literature"])
