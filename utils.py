
import string
import re


def separate_single_multi(l):
    singles, multis = [], []
    for item in l:
        i = item.strip()
        if " " in i:
            multis.append(i)
        else:
            singles.append(i)
    return singles, multis


def verbose(func):
    def inner(*args, **kwargs):
        if kwargs.get("verbose", False):
            i = kwargs.get("indent_level", 0)
            indent = "\t"*i
            print(f"{indent}{args[0].strip()}")
        return func(*args, **kwargs)
    return inner


class AnswerMapping:

    @staticmethod
    @verbose
    def get_numbered_list_items(output, verbose=False, indent_level=0):
        final = []
        if "\n" in output:
            candidates = output.split("\n")
            for cand in candidates:
                c = cand.strip()
                if c.lower().strip() in ["", "answer:"]:
                    pass
                elif re.match(r"\d+[.)]+ *", cand):
                    final.append(cand[2:].strip())
                else:
                    print(f"Unable to match nonempty {c}")
                    pass
        else:
            candidates = re.split(r"\d+[.)]", output)
            for cand in candidates:
                c = cand.strip()
                if c.lower().strip() in ["", "answer:"]:
                    pass
                else:
                    final.append(cand)
        return final

    @staticmethod
    @verbose
    def get_true_or_false(output, default=True, verbose=False, indent_level=0):
        output = output.lower()
        true_condition = "yes " in output or "yes." in output or "true" in output
        false_condition = "no " in output or "no." in output or "false" in output

        if true_condition and not false_condition:
            return True
        elif false_condition and not true_condition:
            return False
        else:
            if not true_condition and not false_condition:
                print(f"Unable to map {output} to True or False")
            else:
                print(f"Mapping {output} to both True or False")
            return default

    @staticmethod
    @verbose
    def exemplar_format_list(output, verbose=False, indent_level=0, separator='|'):
        if "\n" in output:
            listed = AnswerMapping.get_numbered_list_items(output, verbose=verbose, indent_level=indent_level+1)
        else:
            listed = []
            split = re.split(r"\d+[.)]", output)
            for item in split:
                if item.strip().lower() == "" or "answer" in item.strip().lower():
                    pass
                else:
                    listed.append(item.strip())
        final = []
        for option in listed:
            split = option.split(separator)
            if len(split) == 1:
                print(f"Got only one value for {option} with separator '{separator}'")
                continue
            elif len(split) == 2:
                entity, status = split
            elif len(split) == 3:
                entity, status, explanation = split
            else:
                entity, status = split[0], split[1]
                print(f"Got more than 3 values for {option} with separator '{separator}'")
            if status.strip().lower() == "true":
                final.append(entity.strip().lower())
            else:
                pass
        return final
