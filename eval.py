import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm


def is_eq(e1, e2):
    return e1.lower() == e2.lower()


def f1(true_list, pred_list):
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


def eval_dataset(val, model, algorithm):
    algorithm.set_model_fn(model)
    f1s = []
    for i in tqdm(range(len(val))):
        q = val.loc[i]
        para = q['text']
        entities = q['entities']
        algorithm.set_para(para)
        preds = algorithm.perform(verbose=True)
        f1_score = f1(entities, preds)
        f1s.append(f1_score)
    f1s = np.array(f1s)
    return f1s.mean(), f1.std()


def eval_conll(model, algorithm, n_runs=3):
    config = ConllConfig()
    config.set_config(algorithm)
    conll = load_conll2003("validation")
    f1_means, f1_stds = [], []
    for i in range(n_runs):
        f1_mean, f1_std = eval_dataset(conll, model, algorithm)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)
    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_std)
    return f1_means, f1_stds

if __name__ == "__main__":
    from models import T5
    model = T5()
    x, y = eval_conll(model, Algorithm(), n_runs=1)
    print(x)
    print(y)