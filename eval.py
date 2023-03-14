import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm
import time
import pandas as pd
import openai


def is_eq(e1, e2):
    return e1.lower() == e2.lower()


def f1(true_list, pred_list):
    true_list = list(set(true_list))
    pred_list = list(set(pred_list))
    tp = 0
    fn = 0
    fp = 0
    if len(true_list) == 0:
        if len(pred_list) != 0:
            return 0, tp, len(pred_list), fn
        else:
            return 1, tp, fp, fn
    if len(pred_list) == 0:
        if len(true_list) != 0:
            return 0, tp, fp, len(true_list)
        else:
            return 1, tp, fp, fn
    for positive in true_list:
        flag = False
        for candidate in pred_list:
            if is_eq(positive, candidate):
                tp += 1
                flag = True
                break
        if not flag:
            fn += 1
    for candidate in pred_list:
        flag = False
        for positive in true_list:
            if is_eq(positive, candidate):
                flag=True
                break
        if not flag:
            fp += 1
    f1_score = tp/(tp + 0.5*(fp+fn))
    return f1_score, tp, fp, fn


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
                preds, metadata = algorithm.perform(verbose=False)
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


def eval_conll(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True, **kwargs):
    config = ConllConfig()
    algorithm.split_phrases = True
    config.set_config(algorithm, exemplar=exemplar, coT=coT)
    conll = load_conll2003("validation")
    return complete_eval(conll, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_genia(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True, **kwargs):
    config = GeniaConfig()
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT)
    genia = load_genia()
    return complete_eval(genia, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_cross_ner(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                   **kwargs):
    cats = ['politics', 'literature', 'ai', 'science', 'music']
    confs = [CrossNERPoliticsConfig(), CrossNERLiteratureConfig(), CrossNERAIConfig(),
             CrossNERNaturalSciencesConfig(), CrossNERMusicConfig()]
    category = kwargs.get('add_info', None)
    assert category in cats
    i = cats.index(category)
    config = confs[i]
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT)
    dataset = load_cross_ner(category=category)
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_few_nerd_intra(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        **kwargs):
    splits = ["train", "dev", "test"]
    confs = [FewNERDINTRATrainConfig(), FewNERDINTRADevConfig(), FewNERDINTRATestConfig()]
    split = kwargs.get("add_info")
    assert split in splits
    i = splits.index(split)
    config = confs[i]
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT)
    dataset = load_few_nerd(category="intra", split=split)
    return complete_eval(dataset, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def run(dataset="conll", subdataset=None, gpt=False, exemplar=True, coT=True,  name_meta=""):
    res_path = "results"
    gpt_limit = 300
    gpt_nruns = 2
    other_limit = 100
    other_nruns = 3
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
        model = GPT3()
        f1_mean, f1_std, micro_f1, mistakes = eval_fn(model.query, Algorithm_class(), n_runs=gpt_nruns,
                                                      sleep_between_queries=model.seconds_per_query,
                                                      limit=gpt_limit,
                                                      exemplar=exemplar, coT=coT, add_info=subdataset)
    else:
        model = T5XL(size='xxl')
        f1_mean, f1_std, micro_f1, mistakes = eval_fn(model.query, Algorithm_class(), n_runs=other_nruns,
                                                      sleep_between_queries=None, exemplar=exemplar,
                                                      coT=coT, limit=other_limit, add_info=subdataset)
    print(f"Final Results For {dataset} | {subdataset} | {cot} | {exemplar}")
    print(f"f1_means: {f1_mean}")
    print(f"f1_stds: {f1_std}")
    print(f"micro_f1s: {micro_f1}")
    print(f"Saving file to {res_path}/{name_meta}{model.__class__.__name__}_{dataset}{subdataset}.csv")
    mistakes.to_csv(f"{res_path}/{name_meta}{model.__class__.__name__}_{dataset}{subdataset}.csv")


if __name__ == "__main__":
    from models import T5, GPT3, T5XL
    for cot in [True, False]:
        for exemplar in [True, False]:
            run(gpt=True, dataset="conll", coT=cot, exemplar=exemplar, subdataset=f"cot_{cot}_exemplar_{exemplar}")
    for category in ['politics', 'literature', 'ai', 'science', 'music']:
        run(gpt=True, dataset="crossner", coT=True, exemplar=True, subdataset=category)
    for split in ["train", "dev", "test"]:
        run(gpt=True, dataset="fewnerd", coT=True, exemplar=True, subdataset=category)
