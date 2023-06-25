import os
import pandas as pd
from datasets import load_dataset
import re
import string

data_root = "data"


def get_row(func):
    def infunc(frame, i=None):
        if i is not None:
            frame = frame.loc[i, :]
        return func(frame, i=None)
    return infunc


def read_ob2(file_path):
    with open(file_path) as file:
        lines = file.readlines()
    sentences = []
    entities = []
    types = []
    exact_types = []
    data = []
    sub_entities = []
    sub_types = {}
    sub_exact_types = []
    words = ""
    curr_entity = ""
    curr_type = None

    for i, line in enumerate(lines):
        if line.strip() == "" or line == "\n" or i == len(lines)-1:
            # save entity if it exists
            if curr_type is not None:
                sub_entities.append(curr_entity.strip())
                sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            if words != "":
                sentences.append(words)
                entities.append(sub_entities)
                types.append(sub_types)
                exact_types.append(sub_exact_types)
                data.append([words, sub_entities, sub_types, sub_exact_types])
            sub_entities = []
            sub_types = {}
            sub_exact_types = []
            words = ""
            curr_entity = ""
            curr_type = None
        else:
            word, tag = line.split("\t")
            if words == "":
                words = word
            else:
                words = words + " " + word
            sub_exact_types.append(tag.strip())
            if tag.split() == "O" or "-" not in tag:  # if there was an entity before this then add it in full
                if curr_type is not None:
                    sub_entities.append(curr_entity.strip())
                    sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            elif "B-" in tag or "I-" in tag:
                if "B-" in tag:
                    if curr_type is not None:
                        sub_entities.append(curr_entity.strip())
                        sub_types[curr_entity.strip()] = curr_type
                    curr_entity = word
                    curr_type = tag.split("-")[1].strip()
                else:  # I- in tag
                    if curr_type is None:
                        print(f"Should not be happening bug here")
                    curr_entity = curr_entity + " " + word
            else:
                main_type, subtype = tag.split("-")  # must assume that if curr_type is not None then its the same one because FewNERD doesn't contain B, I information
                if subtype.strip() == "government/governmentagency":
                    subtype = "government"
                if curr_type is None:
                    curr_entity = word
                    curr_type = main_type + "-" + subtype.strip()  # can change to make it subtype if we want
                else:
                    curr_entity = curr_entity + " " + word

    df = pd.DataFrame(columns=["text", "entities", "types", "exact_types"], data=data)
    return df


def write_ob2(df, dataset_folder=None, filename=None):
    assert dataset_folder is not None and filename is not None
    os.makedirs(data_root + "/" + dataset_folder, exist_ok=True)
    with open(data_root + "/"+ dataset_folder +"/"+filename+".txt", "w") as f:
        for i in df.index:
            row = df.loc[i]
            sentence = row["text"]
            types = row["exact_types"]
            for j, word in enumerate(sentence.split(" ")):
                f.write(f"{word}\t{types[j]}\n")
            f.write("\n")
    return


def load_conll2003(split="validation"):
    dset = load_dataset("conll2003")[split]
    columns = ["text", "entities", "types", "exact_types"]
    #'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    conll_tag_map = {0: "none", 1: "per", 2: "per", 3: "org", 4: "org", 5: "loc", 6: "loc", 7: "misc", 8: "misc"}
    conll_fulltagmap = {0: "O", 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    data = []
    for j in range(len(dset)):
        text = " ".join(dset[j]['tokens'])
        types = dset[j]["ner_tags"]
        sentence = text.split(" ")
        assert len(sentence) == len(types)
        entities = []
        d = {}
        subentities = ""
        curr_type = None
        exacts = []
        for i, tag in enumerate(types):
            exacts.append(conll_fulltagmap[tag])
            if tag == 0:
                if curr_type is not None:
                    entities.append(subentities)
                    d[subentities] = curr_type
                    curr_type = None
                    subentities = ""
            else:
                if tag in [1, 3, 5, 7]:
                    if curr_type is not None:
                        entities.append(subentities)
                        d[subentities] = curr_type
                    curr_type = conll_tag_map[tag]
                    subentities = sentence[i]
                else:
                    assert curr_type is not None
                    subentities = subentities + " " + sentence[i]
        data.append([text, entities, d, exacts])
    df = pd.DataFrame(columns=columns, data=data)
    return df


def load_ontonotes(split="validation", save_ob2=True):
    dset = load_dataset("conll2012_ontonotesv5", 'english_v4')[split]
    columns = ["text", "entities", "types", "exact_types"]
    onto_tags = ["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"]
    onto_tag_map = {}
    for i in range(len(onto_tags)):
        onto_tag_map[i] = onto_tags[i]
    onto_fulltagmap = onto_tag_map
    data = []
    for k in range(len(dset)):
        for j in dset[k]['sentences']:
            text = " ".join(dset[j]['words'])
            types = dset[j]["named_entities"]
            sentence = text.split(" ")
            assert len(sentence) == len(types)
            entities = []
            d = {}
            subentities = ""
            curr_type = None
            exacts = []
            for i, tag in enumerate(types):
                exacts.append(onto_fulltagmap[tag])
                if tag == 0:
                    if curr_type is not None:
                        entities.append(subentities)
                        d[subentities] = curr_type
                        curr_type = None
                        subentities = ""
                else:
                    if tag % 2 == 1:  # then it is a B
                        if curr_type is not None:
                            entities.append(subentities)
                            d[subentities] = curr_type
                        curr_type = onto_tag_map[tag]
                        subentities = sentence[i]
                    else:
                        assert curr_type is not None
                        subentities = subentities + " " + sentence[i]
            data.append([text, entities, d, exacts])
    df = pd.DataFrame(columns=columns, data=data)
    if save_ob2:
        if split == "validation":
            split = "dev"
        write_ob2(df, dataset_folder="ontoNotes", filename=f"{split}")
    return df


def load_genia(genia_path="data/Genia/Genia4ERtask1.iob2"):
    return read_ob2(genia_path)


def load_few_nerd(few_nerd_path="data/FewNERD", category="intra", split="test"):
    assert category in ["inter", "intra", "supervised"]
    file_path = os.path.join(few_nerd_path, category, f"{split}.txt")
    return read_ob2(file_path)


def load_cross_ner(cross_ner_path='data/CrossNER', category="ai"):
    assert category in ['politics', 'literature', 'ai', 'science', 'conll2003', 'music']
    file_path = os.path.join(cross_ner_path, "ner_data", category, "dev.txt")
    return read_ob2(file_path)


def scroll(dataset, start=0, exclude=None):
    cols = dataset.columns
    for i in range(start, len(dataset)):
        s = dataset.loc[i]
        print(f"Item: {i}")
        for col in cols:
            if exclude is not None:
                if col in exclude:
                    continue
            print(f"{col}")
            print(s[col])
            print(f"XXXXXXXXXXXXXXX")
        inp = input("Continue?")
        if inp != "":
            return
