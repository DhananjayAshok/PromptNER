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
    if len(true_list) == 0:
        if len(pred_list) != 0:
            return 0
        else:
            return 1
    if len(pred_list) == 0:
        if len(true_list) != 0:
            return 0
        else:
            return 1
    tp = 0
    fn = 0
    fp = 0
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
    tp, fp, fn = 0
    mistake_data = []
    for i in tqdm(range(len(val))):
        q = val.loc[i]
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
        f1_score, tp, fp, fn = f1(entities, preds)
        if f1_score != 1:
            mistake_data.append([i, para, entities, preds, metadata, f1_score])
        f1s.append(f1_score)
        if print_every is not None:
            if i % print_every == 0:
                avg_f1 = np.array(f1s).mean()
                std_f1 = np.array(f1s).std()
                micro_f1 = tp/(tp + 0.5*(fp+fn))
                print(f"Iteration {i}: Avg f1: {avg_f1}, Std f1: {std_f1}, micro f1: {micro_f1}")
    f1s = np.array(f1s)
    micro_f1 = tp / (tp + 0.5 * (fp + fn))
    return f1s.mean(), f1s.std(), micro_f1, mistake_data


def complete_eval(dataset, model, algorithm, n_runs=3, sleep_between_queries=None, limit=None):
    f1_means, f1_stds, micro_f1s = [], [], []
    mistake_columns = ["idx", "para", "entities", "preds", "meta", "f1"]
    if limit is not None:
        dataset = dataset.sample(limit)
    for i in range(n_runs):
        f1_mean, f1_std, micro_f1, mistake_data = eval_dataset(dataset, model, algorithm, sleep_between_queries=sleep_between_queries)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)
        micro_f1s.append(micro_f1)
    df = pd.DataFrame(data=mistake_data, columns=mistake_columns)
    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_std)
    micro_f1s = np.array(micro_f1s)
    return f1_means, f1_stds, micro_f1s, df


def eval_conll(model, algorithm, n_runs=3, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
               generic=False):
    config = ConllConfig()
    algorithm.split_phrases = True
    config.set_config(algorithm, exemplar=exemplar, coT=coT, generic=generic)
    conll = load_conll2003("validation")
    return complete_eval(conll, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


def eval_genia(model, algorithm, n_runs=3, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
               generic=False):
    config = GeniaConfig()
    algorithm.split_phrases = True
    config.set_config(algorithm, exemplar=exemplar, coT=coT, generic=generic)
    genia = load_genia()
    return complete_eval(genia, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)


if __name__ == "__main__":
    from models import T5, GPT3, T5XL
    res_path = "results"
    gpt_limit = 500
    other_limit = 500
    Algorithm_class = Algorithm

    gpt = True
    exemplar = True
    coT = True
    generic = False
    dataset = "conll"
    name_meta = ""
    modes = [1]

    if dataset == "conll":
        eval_fn = eval_conll
    elif dataset == "genia":
        eval_fn = eval_genia
    for mode in modes:
        if gpt:
            model = GPT3()
            sleep_between_queries = model.seconds_per_query
            f1_mean, f1_std, micro_f1, mistakes = eval_fn(model.query, Algorithm_class(mode=mode), n_runs=1,
                                     sleep_between_queries=model.seconds_per_query, limit=gpt_limit,
                                                          exemplar=exemplar, coT=coT, generic=generic)
        else:
            model = T5XL(size='xxl')
            f1_mean, f1_std, micro_f1, mistakes = eval_fn(model.query, Algorithm_class(mode=mode), n_runs=1,
                                                          sleep_between_queries=None, exemplar=exemplar,
                                                          coT=coT, limit=other_limit, generic=generic)

        print(f"f1_means: {f1_mean}")
        print(f"f1_stds: {f1_std}")
        print(f"micro_f1s: {micro_f1}")
        print(f"Saving file to {res_path}/{name_meta}{model.__class__.__name__}_{mode}_{dataset}.csv")
        mistakes.to_csv(f"{res_path}/{name_meta}{model.__class__.__name__}_{mode}_{dataset}.csv")
