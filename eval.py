import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm
import time
import pandas as pd

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
    return f1_score


def eval_dataset(val, model, algorithm, sleep_between_queries=None):
    algorithm.set_model_fn(model)
    f1s = []
    mistake_data = []
    for i in tqdm(range(len(val))):
        q = val.loc[i]
        para = q['text']
        entities = q['entities']
        algorithm.set_para(para)
        if sleep_between_queries is not None:
            time.sleep(sleep_between_queries)
        print(entities)
        preds = algorithm.perform(verbose=True)
        f1_score = f1(entities, preds)
        if f1_score != 1:
            mistake_data.append([i, para, entities, preds, f1_score])
        f1s.append(f1_score)
    f1s = np.array(f1s)
    return f1s.mean(), f1s.std(), mistake_data


def eval_conll(model, algorithm, n_runs=3, sleep_between_queries=None, limit=None):
    config = ConllConfig()
    config.set_config(algorithm)
    conll = load_conll2003("validation").loc[:limit]
    f1_means, f1_stds = [], []
    mistake_columns = ["idx", "para", "entities", "preds", "f1"]
    for i in range(n_runs):
        f1_mean, f1_std, mistake_data = eval_dataset(conll, model, algorithm, sleep_between_queries=sleep_between_queries)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)
    df = pd.DataFrame(data=mistake_data, columns=mistake_columns)
    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_std)
    return f1_means, f1_stds, df


if __name__ == "__main__":
    from models import T5, GPT3
    model = GPT3
    x, y, mistakes = eval_conll(model.query, Algorithm(), n_runs=1, sleep_between_queries=model.seconds_per_query,
                                limit=20)
    print(x)
    print(y)