import os
import pandas as pd
from datasets import load_dataset
import re
import string

data_root = "data"
re_root = os.path.join(data_root, "RelationExtraction")
re_options = os.listdir(re_root)


def get_row(func):
    def infunc(frame, i=None):
        if i is not None:
            frame = frame.loc[i, :]
        return func(frame, i=None)
    return infunc


def get_re_dset(name=None, i=None):
    if (name is None and i is None) or (name is not None and name not in re_options) or \
            (i is not None and i > len(re_options)):
        name = re_options[0]
    elif i is not None:
        name = re_options[i]
    train = pd.read_json(os.path.join(re_root, name, "train.json"), lines=True)
    def smap(x):
        s = ""
        for iii in x:
            s = s + iii + " "
        return s
    train['token'] = train['token'].map(smap)
    return train


def read_ob2(file_path):
    with open(file_path) as file:
        lines = file.readlines()
    sentences = []
    entities = []
    entity_types = []
    entity_type_match = []

    entity_list = []
    type_list = []
    working_sentence = ""
    working_entity = ""
    for line in lines:
        if line == "\n":
            if working_sentence != "":
                sentences.append(working_sentence)
                if working_entity != "":
                    entity_list.append(working_entity)
                entities.append(entity_list)
                entity_types.append(type_list)
            entity_list = []
            working_sentence = ""
            working_entity = ""
            type_list = []
        l = line.strip().split("\t")
        if len(l) == 0 or l[0].strip() == "":
            continue
        if working_sentence == "":
            working_sentence = l[0]
        else:
            working_sentence = working_sentence + " " + l[0]
        type_list.append(l[1])
        if len(l) != 2 or l[1] == "O":
            if working_entity != "":
                entity_list.append(working_entity)
                working_entity = ""
        else:
            if l[1][0] == "B":
                if working_entity != "":
                    # Assume that its two different entities
                    entity_list.append(working_entity)
                working_entity = l[0]
            elif l[1][0] == "I":
                if working_entity == "":
                    print(f"Theres a problem here no working entity, new entity {l[1][0]}. Line {line}")
                working_entity = working_entity + " " + l[0]
    columns = ["text", "entities", "types"]
    data = []
    for i in range(len(sentences)):
        tokens = sentences[i].split(" ")
        ts = entity_types[i]
        d = {}
        for j, token in enumerate(tokens):
            if token in entities[i]:
                d[token] = ts[j]
        data.append([sentences[i], entities[i], d])
    df = pd.DataFrame(columns=columns, data=data)
    return df


def load_conll2003(split="validation"):
    dset = load_dataset("conll2003")[split]
    columns = ["text", "entities", "types"]
    data = []
    for i in range(len(dset)):
        text = " ".join(dset[i]['tokens'])
        types = dset[i]["ner_tags"]
        sentence = text.split(" ")
        assert len(sentence) == len(types)
        entities = []
        d = {}
        for i, tag in enumerate(types):
            if tag != 0:
                entities.append(sentence[i])
                d[sentence[i]] = tag
        data.append([text, entities, d])
    df = pd.DataFrame(columns=columns, data=data)
    return df


def load_genia(genia_path="data/Genia/Genia4ERtask1.iob2"):
    return read_ob2(genia_path)
