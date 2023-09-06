import os

from data import *
import pandas as pd
import numpy as np
import string
from utils import AnswerMapping
from algorithms import BaseAlgorithm, Algorithm
from data import scroll
from models import OpenAIGPT
import warnings
import random
from seqeval.metrics import f1_score as seq_f1
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
results_dir = "results"



def is_eq(e1, e2):
    return e1.lower().strip().strip(string.punctuation).strip() == e2.lower().strip().strip(string.punctuation).strip()


def basic_process_results(filename):
    df = pd.read_csv(results_dir+"/"+filename)
    for col in ["entities", "truth", "pred"]:
        df[col] = df[col].apply(eval)
    df['pred_text'] = None
    df['truth_text'] = None
    df['correct'] = df['pred'] == df['truth']
    for i in df.index:
        row = df.loc[i]
        text = row['text'].split(" ")
        preds = row['pred']
        truths = row['truth']
        pred_text = ""
        truth_text = ""
        for j, word in enumerate(text):
            if pred_text == "":
                pred_text = word + " | " + preds[j]
            else:
                pred_text = pred_text + " " + word + " | " + preds[j]
            if truth_text == "":
                truth_text = word + " | " + truths[j]
            else:
                truth_text = truth_text + " " + word + " | " + truths[j]
        df.loc[i, "pred_text"] = pred_text
        df.loc[i, "truth_text"] = truth_text
    df.to_csv(results_dir + "/"+filename, index=False)
    return


def process_all_results():
    for filename in os.listdir(results_dir):
        basic_process_results(filename)


def workbench():
    d = {}
    for file in os.listdir(results_dir):
        d[file] = pd.read_csv(results_dir+"/"+file)
        for col in ["entities", "truth", "pred"]:
            d[file][col] = d[file][col].apply(eval)
    formatter = lambda x: AnswerMapping.exemplar_format_list(x, identify_types=True, verbose=False)
    alg = Algorithm(model_fn=OpenAIGPT.query)

    def do_span(para, meta):
        alg.set_para(para)
        answers, typestrings = formatter(meta)
        return alg.parse_span(answers, typestrings, meta)
    return d, formatter, do_span


def analytics(d):
    d["text_len"] = d['text'].apply(lambda x: len(x.split(" ")))
    d["n_entities"] = d['entities'].apply(len)
    all_types = []
    for i in d.index:
        types = list(set(d.loc[i, "truth"]))
        all_types.extend(types)
    all_types = list(set(all_types))
    all_types.sort()
    truths = []
    preds = []
    for i in d.index:
        row = d.loc[i]
        truth = row["truth"]
        pred = row["pred"]
        truths.extend(truth)
        preds.extend(pred)
    print(f"Correlation is: ")
    print(d.corr()["f1"])
    conf = confusion_matrix(truths, preds, labels=all_types)
    return conf











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

    return batch


def connect_turk_output(turk_name="survey_result", save_name="survey_data", n_workers=10):
    all_batches = pd.concat([process_batch(turk_name, worker) for worker in range(n_workers)], ignore_index=True)
    df = pd.read_csv(f"results/survey/{save_name}.csv")
    return df, all_batches


def process_batch_row(row):
    gptno = row['gptlist']
    trueno = 1 if gptno == 2 else 2
    gptcorrect = int(row[f"l{gptno}correct"])
    truecorrect = int(row[f"l{trueno}correct"])

    gptbetter = int(row["better"] == gptno)
    gptworse = int(row["better"] == trueno)
    if row[f"l{gptno}missing"] is not None and not isinstance(row[f"l{gptno}missing"], float):
        gptmissing = len(row[f"l{gptno}missing"].split(","))
    else:
        gptmissing = 0
    if row[f"l{gptno}extra"] is not None and not isinstance(row[f"l{gptno}extra"], float):
        gptextra = len(row[f"l{gptno}extra"].split(","))
    else:
        gptextra = 0
    if row[f"l{trueno}missing"] is not None and not isinstance(row[f"l{trueno}missing"], float):
        truemissing = len(row[f"l{trueno}missing"].split(","))
    else:
        truemissing = 0
    if row[f"l{trueno}extra"] is not None and not isinstance(row[f"l{trueno}extra"], float):
        trueextra = len(row[f"l{trueno}extra"].split(","))
    else:
        trueextra = 0
    return gptcorrect, truecorrect, gptbetter, gptworse, gptmissing, gptextra, truemissing, trueextra


def summarize(df, i, column, array):
    df.loc[i, f"{column}"] = pd.Series(array).value_counts().index[0]
    avg = sum(array) / len(array)
    if avg == pd.Series(array).value_counts().index[0]:
        df.loc[i, f"{column}_agreement"] = 1
    elif len(array) == 2:
        df.loc[i, f"{column}_agreement"] = 0
    else:
        df.loc[i, f"{column}_agreement"] = 0.5
    return


def process_turk():
    df, all_batches = connect_turk_output()
    for i in df.index:
        subset = all_batches[all_batches['id'] == i]
        df.loc[i, "num"] = len(subset)
        gptcorrects, truecorrects, gptbetters, gptworses, gptmissings, gptextras, truemissings, trueextras = [], [], \
            [], [], [], [], [], []
        for j in subset.index:
            gptcorrect, truecorrect, gptbetter, gptworse, gptmissing, gptextra, truemissing, trueextra = \
                process_batch_row(subset.loc[j])
            gptcorrects.append(gptcorrect)
            truecorrects.append(truecorrect)
            gptbetters.append(gptbetter)
            gptworses.append(gptworse)
            gptmissings.append(gptmissing)
            gptextras.append(gptextra)
            truemissings.append(truemissing)
            trueextras.append(trueextra)
        summarize(df, i, "gptcorrect", gptcorrects)
        summarize(df, i, "truecorrect", truecorrects)
        summarize(df, i, "gptbetter", gptbetters)
        summarize(df, i, "gptworse", gptworses)
        df.loc[i, "gptmissing"] = sum(gptmissings) / len(gptmissings)
        df.loc[i, "gptextra"] = sum(gptextras) / len(gptextras)
        df.loc[i, "truemissing"] = sum(truemissings) / len(truemissings)
        df.loc[i, "trueextra"] = sum(trueextras) / len(trueextras)
    df.to_csv(f"results/survey/final_results.csv")
    return df


def analyze_turk():
    df = process_turk()
    slices = [(df, "All")]
    for dataset in df['dataset'].unique():
        slices.append((df[df['dataset'] == dataset], dataset))
    for slice, name in slices:
        print(f"For {name}")
        print(f"\tGPTCorrect: {slice['gptcorrect'].mean()}, Agreement: {slice['gptcorrect_agreement'].mean()}")
        print(f"\tTrueCorrect: {slice['truecorrect'].mean()}, Agreement: {slice['truecorrect_agreement'].mean()}")
        print(f"\tGPTBetter: {slice['gptbetter'].mean()}, Agreement: {slice['gptbetter_agreement'].mean()}")
        print(f"\tGPTWorse: {slice['gptworse'].mean()}, Agreement: {slice['gptworse_agreement'].mean()}")
        print(f"\tAvg GPT Missing and Extra: {slice['gptmissing'].mean()}, {slice['gptextra'].mean()}")
        print(f"\tAvg True Missing and Extra: {slice['truemissing'].mean()}, {slice['trueextra'].mean()}")


if __name__ == "__main__":
    process_all_results()


