from data import *
import pandas as pd
import numpy as np
import string
from utils import AnswerMapping
from algorithms import BaseAlgorithm
from data import scroll
import random

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


def bulk_eval(filenames=[ "GPT4QOpenAIGPT_conllNone.csv", "GPT4QOpenAIGPT_crossnerai.csv",
                          "GPT4QOpenAIGPT_crossnerliterature.csv", "GPT4QOpenAIGPT_crossnermusic.csv",
                          "GPT4QOpenAIGPT_crossnerpolitics.csv", "GPT4QOpenAIGPT_crossnerscience.csv",
                          "GPT4QOpenAIGPT_fewnerdtest.csv"],
              keys=["conll", "ai", "lit", "music", "pol", "science", "fewnerd"]):
    dfs = {}
    for i, filename in enumerate(filenames):
        split_phrases = "conll" in filename
        df = get_results_frame(filename, split_phrases=split_phrases)
        dfs[keys[i]] = df
    return dfs


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


def_pre = "Named entities are phrases that represent the name of a "
defn_map = {'conll': "person, organization or location",
            'ai': "field, task, product, algorithm, researcher, metrics, university, country, person, organization or location",
            'lit': "book, writer, award, poem, event, magazine, person, location, organization, country, miscellaneous",
            'music': "music genre, song, band, album, musical artist, musical instrument, award, event, country, location, organization or person",
            'pol': "politician, person, organization, political party, event, election, country or location",
            'science': "scientist, person, university, organization, country, location, discipline, enzyme, protein, chemical compound, chemical element, event, astronomical object, academic journal, award or theory"}


def get_survey_format(dfs, save_name="survey_data", examples_per_dataset=20, n_attentions=2, n_workers=10, n_examples_per_worker=30):
    columns = ["defn", "sentence", "list1", "list2", "gptlist", "f1", "dataset"]
    data = []
    for dataset in dfs:
        if dataset in ["fewnerd", "conll"]:
            continue
        df = dfs[dataset]
        defn = def_pre + defn_map[dataset]
        for i in df.index:
            f1 = df.loc[i, "f1"]
            if f1 == 1:
                keep = random.random()
                if keep < 0.85:
                    continue
            sentence = df.loc[i, "para"]
            pred = df.loc[i, "preds"]
            true = df.loc[i, "entities"]
            pred = list(set(pred))
            true = list(set(true))
            np.random.shuffle(pred)
            np.random.shuffle(true)
            pred = ", ".join(pred)
            true = ", ".join(true)
            if len(true) == 0:
                continue
            flip = random.random()
            if flip > 0.5:
                gptlist = 1
                list1 = pred
                list2 = true
            else:
                gptlist = 2
                list1 = true
                list2 = pred
            mdata = [defn, sentence, list1, list2, gptlist, f1, dataset]
            data.append(mdata)
    df = pd.DataFrame(columns=columns, data=data)
    attentions = df[df.f1 == 1].reset_index(drop=True)
    attentions['id'] = -1
    sample_dfs = []
    for dataset in df['dataset'].unique():
        samples = df[(df.dataset == dataset) & (df.f1 != 1)].sample(examples_per_dataset).reset_index(drop=True)
        sample_dfs.append(samples)
    df = pd.concat(sample_dfs, ignore_index=True)
    df = df.sample(len(df)).reset_index(drop=True)
    df['id'] = df.index
    save_total = f"results/survey/{save_name}"
    df.to_csv(f"{save_total}.csv", index=False)
    n_examples = len(df)
    workers_per_example = (n_examples_per_worker * n_workers) // n_examples
    workers = list(range(n_workers))
    split_dfs = [pd.DataFrame(columns=df.columns, data=[]) for i in workers]
    for i in df.index:
        selected = np.random.choice(workers, workers_per_example, replace=False)
        for sel in selected:
            split_dfs[sel] = pd.concat([split_dfs[sel], df.loc[i:i]], ignore_index=True)
            if len(split_dfs[sel]) >= n_examples_per_worker:
                workers.remove(sel)
            if len(workers) < workers_per_example:
                workers = list(range(n_workers))
    for i, split_df in enumerate(split_dfs):
        split_dfs[i] = pd.concat([split_df, attentions.sample(n_attentions)], ignore_index=True)

        print(f"worker: {i} assigned {len(split_dfs[i])} examples")
        split_dfs[i].sample(len(split_dfs[i])).reset_index(drop=True).to_csv(f"{save_total}_{i}.csv", index=False)
    return df, split_dfs


def gen_survey_format():
    dfs = bulk_eval()
    df, split_dfs = get_survey_format(dfs)
    return df, split_dfs


def process_batch(turk_name="batch", worker=0):
    batch = pd.read_csv(f"results/survey/{turk_name}_{worker}.csv")
    batch.drop(['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward',
       'CreationTime', 'MaxAssignments', 'RequesterAnnotation',
       'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds',
       'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds',
       'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
       'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
       'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
       'Last30DaysApprovalRate', 'Last7DaysApprovalRate'], axis=1, inplace=True)
    inputs = []
    answers = []
    for column in batch:
        if "Input" in column:
            inputs.append(column)
        elif "Answer" in column:
            answers.append(column)

    for inp in inputs + answers:
        col = inp.split(".")[1]
        batch[col] = batch[inp]
        batch.drop(inp, axis=1, inplace=True)

    if "l1add" not in batch.columns:
        batch['l1add'] = None

    if "l2add" not in batch.columns:
        batch["l2add"] = None

    return batch


def connect_turk_output(turk_name="batch", save_name="survey_data", n_workers=10):
    all_batches = pd.concat([process_batch(turk_name, worker) for worker in range(n_workers)], ignore_index=True)
    df = pd.read_csv(f"results/survey/{save_name}.csv")
    return df, all_batches


