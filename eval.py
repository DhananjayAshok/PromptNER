from data import *
import pandas as pd
import string
from utils import AnswerMapping
from algorithms import BaseAlgorithm


def fn(x, pred_col="preds"):
    preds = set(x[pred_col])
    entities = set(x["entities"])
    return list(entities.difference(preds))


def fp(x, pred_col="preds"):
    preds = set(x[pred_col])
    entities = set(x["entities"])
    return list(preds.difference(entities))


def generous_inclusion(x, pred_col="preds"):
    incls = []
    for item in x["entities"]:
        flag = False
        for pred in x[pred_col]:
            if item in pred:
                flag = True
                break
        if not flag:
            incls.append(item)
    return incls


def lower_map(x):
    return [i.lower() for i in x]


def split(x):
    new_l = []
    for item in x:
        new_l.extend(item.split(" "))
    return new_l


def get_results_frame(filename, results_dir="results", split_phrases=False, clean_output=True):
    if ".csv" not in filename:
        filename = filename + ".csv"
    df = pd.read_csv(results_dir+"/"+filename)
    df["preds"] = df["preds"].apply(eval).apply(lower_map)
    df["entities"] = df["entities"].apply(eval).apply(lower_map)
    df["fn"] = df.apply(fn, axis=1)
    df["fp"] = df.apply(fp, axis=1)
    df["gen_fn"] = df.apply(generous_inclusion, axis=1)
    df["candidates"] = df.apply(lambda x: AnswerMapping.exemplar_format_list(x["meta"], true_only=False), axis=1).apply(lower_map)
    if split_phrases:
        df["candidates"] = df["candidates"].apply(split)
    if clean_output:
        df["candidates"] = df["candidates"].apply(BaseAlgorithm.clean_output)
    df["candidate_fn"] = df.apply(lambda x: fn(x, pred_col="candidates"), axis=1)
    df["candidate_fp"] = df.apply(lambda x: fp(x, pred_col="candidates"), axis=1)
    df["candidate_gen_fn"] = df.apply(lambda x: generous_inclusion(x, pred_col="candidates"), axis=1)
    print(f"Aggregate Analysis on {filename}:")
    print(f"On Predictions")
    print(f"\tFalse Positives (Mean: {df['fp'].apply(len).mean()} Std: {df['fp'].apply(len).std()})")
    print(f"\tFalse Negatives (Mean: {df['fn'].apply(len).mean()} Std: {df['fn'].apply(len).std()})")
    print(f"\tGenerous False Negatives (Mean: {df['gen_fn'].apply(len).mean()} Std: {df['gen_fn'].apply(len).std()})")

    print(f"On Candidates")
    print(f"\tFalse Positives (Mean: {df['candidate_fp'].apply(len).mean()} Std: {df['candidate_fp'].apply(len).std()})")
    print(f"\tFalse Negatives (Mean: {df['candidate_fn'].apply(len).mean()} Std: {df['candidate_fn'].apply(len).std()})")
    print(f"\tGenerous False Negatives (Mean: {df['candidate_gen_fn'].apply(len).mean()} Std: {df['candidate_gen_fn'].apply(len).std()})")
    return df


def bulk_eval():
    filenames = [ "GPT3_conllNone.csv", "GPT3_crossnerai.csv", "GPT3_crossnerliterature.csv",
                  "GPT3_crossnermusic.csv", "GPT3_crossnerpolitics.csv", "GPT3_crossnerscience.csv"]
    for filename in filenames:
        split_phrases = "conll" in filename
        get_results_frame(filename, split_phrases=split_phrases)


def is_eq(e1, e2):
    return e1.lower().strip().strip(string.punctuation).strip() == e2.lower().strip().strip(string.punctuation).strip()


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
