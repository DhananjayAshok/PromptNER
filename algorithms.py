import utils
from utils import AnswerMapping
from nltk.corpus import stopwords



class BaseAlgorithm:
    defn = "An entity is an object, place, individual, being, title, proper noun or process that has a distinct and " \
           "independent existence. The name of a collection of entities is also an entity. Adjectives, verbs, numbers, " \
           "adverbs, abstract concepts are not entities. No phrase that is longer than [WORDLIMIT] words is an entity." \
           "Dates, years and times are not entities"

    # if [] = n then there are O(n^2) phrase groupings

    base_task = "Make a numbered list of all the entities in the paragraph: "
    phrase_entity_task = "Does the phrase or word '[WORD]' represent an entity? Explain Why"

    removal_task = "Which items in this list are not actually entities. " \
                   "Report a numbered list."
    whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                 "and for each entry explain why it either is or is not an entity. \nParagraph:"

    para_reference = "Given the following paragraph: "
    phrase_iter = "For every phrase in this list "
    phrase_iter1 = "Go over this list and for each item "
    boolean_force = "Answer with 'True or 'False''"

    def __init__(self, model_fn=None, word_limit=4, split_phrases=False):
        self.defn = self.defn.replace("[WORDLIMIT]", f"{word_limit}")
        self.para = None
        self.model_fn = model_fn
        self.split_phrases = split_phrases

    def set_para(self, para):
        self.para = para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    def initial_query(self, verbose=False, indent_level=0):
        q = self.defn + "\n" + self.para_reference + "\n" + self.para + "\n" + self.base_task
        initial_list = AnswerMapping.get_numbered_list_items(self.model_fn(q), verbose=verbose, indent_level=indent_level)
        return initial_list

    def removal_query(self, working_list, verbose=False, indent_level=0):
        list_str = ""
        for i, entity in enumerate(working_list):
            list_str = list_str + f"\n{i}. {entity}"
        task = f"You are given the following list: {list_str}. {self.removal_task}"
        q = self.defn + "\n" + self.para_reference + "\n" + self.para + "\n" + task
        removal_answer = AnswerMapping.get_numbered_list_items(self.model_fn(q), verbose=verbose, indent_level=indent_level)
        return removal_answer

    def check_word(self, word, force=True, verbose=False, indent_level=0):
        task = self.phrase_entity_task.replace("[WORD]", word)
        if force:
            task = task + " " + self.boolean_force
        q = f"{self.defn}\n{self.para_reference}'{self.para}'\n{task}"
        return AnswerMapping.get_true_or_false(self.model_fn(q), verbose=verbose, indent_level=indent_level)

    def check_composition_speciality(self, entity_list, phrase, force=True, verbose=False, indent_level=0):
        task = self.speciality_task.replace("[WORDLIST]", f"{entity_list}").replace("[PHRASE]", phrase)
        if force:
            task = task + " " + self.boolean_force
        q = f"{self.defn}\n{self.para_reference}'{self.para}'\n{task}"
        return AnswerMapping.get_true_or_false(self.model_fn(q), verbose=verbose, indent_level=indent_level)


class Algorithm(BaseAlgorithm):
    def perform(self, mode=1, verbose=True):
        """

        :param model:
        :param paragraph:
        :param mode: 0: most exhaustive, 2: least exhaustive
        :return:
        """
        if mode == 0:
            answers, metadata = self.perform_exhaustive(verbose=verbose)
        elif mode == 1:
            answers, metadata = self.perform_single_query(verbose=verbose)
        else:
            answers = None
        for trivial in ["", " ", "."] + stopwords.words('english'):
            while trivial in answers:
                answers.remove(trivial)
        if self.split_phrases:
            new_answers = []
            for answer in answers:
                if " " not in answer:
                    new_answers.append(answer)
                else:
                    minis = answer.split(" ")
                    for mini in minis:
                        new_answers.append(mini)
            return new_answers, metadata
        else:
            return answers, metadata

    def perform_exhaustive(self, verbose=True):
        initial_list = self.initial_query(verbose=verbose)
        if verbose:
            print(f"Processed initial list is: {initial_list}")
        working_list = []
        for item in initial_list:
            if verbose:
                print(f"Checking Validity of {item}")
            is_entity = self.check_word(item, verbose=verbose, indent_level=1)
            if is_entity:
                working_list.append(item)
            else:
                words = item.split(" ")
                if len(words) > 1:
                    if verbose:
                        print(f"{item} was not an entity, going into words of the phrase")
                    for word in words:
                        is_entity = self.check_word(word, verbose=verbose, indent_level=2)
                        if is_entity:
                            working_list.append(item)
        if verbose:
            print(f"Working List now: {working_list}")
        singles, multiples = utils.separate_single_multi(working_list)
        pure_entity_multiples = []
        mixed_entity_multiples = []
        for multi in multiples:
            flag = True
            if verbose:
                print(f"\tWorking on {multi}")
            words = multi.split(" ")
            for i, word in enumerate(words):
                if verbose:
                    print(f"\t\tChecking {word}")
                is_entity = self.check_word(word, verbose=verbose, indent_level=2)
                if not is_entity:
                    flag = False
                    break
            if flag:
                pure_entity_multiples.append(multi)
            else:
                mixed_entity_multiples.append(multi)
        working_list = pure_entity_multiples + singles + mixed_entity_multiples
        final_list = working_list
        return final_list, None

    def perform_single_query(self, verbose=True):
        exemplar_construction = ""
        exemplar_construction += self.defn + "\n"
        for exemplar in self.exemplars:
            exemplar_construction += self.whole_task + "\n"
            exemplar_construction += exemplar + "\n"
        exemplar_construction += self.whole_task + "\n"
        output = self.model_fn(exemplar_construction + f"'{self.para}'" + "\nAnswer:")
        final = AnswerMapping.exemplar_format_list(output, verbose=verbose)
        return final, output


class ConllConfig:
    defn = "An entity is an object, place, individual, being, title or process that has a distinct and " \
                "independent existence. The name of a collection of entities is also an entity. " \
           "Names, first names, last names, countries and nationalitites are entities " \
                "Sports, adjectives, verbs, numbers, " \
                "adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. " \
           "Possessive words like I, you, him and me are not entities."

    phrase_entity_task = "Does the phrase or word '[WORD]' represent an object, place, individual or title that has " \
                         "a distinct and independant existence. " \
                         "Answer no if the word represents a time, date, name of sport or abstract concept"

    phrase_entity_task_exemplar = """
    Does the phrase or word 'Innings Victory' represent an object, place, individual or title that has a distinct and independant physical existence. Answer no if the word represents a time, date, name of sport or abstract concept
    Answer: No. Innings victory is an abstract concept of winning an innings which does not have a distinct and independant physical existence
    
    Does the phrase or word 'Grace Road' represent an object, place, individual or title that has a distinct and independant physical existence. Answer no if the word represents a time, date, name of sport or abstract concept
    Answer: Yes. Grace Road is the name of a place. 
    
    Does the phrase or word 'England' represent an object, place, individual or title that has a distinct and independant physical existence. Answer no if the word represents a time, date, name of sport or abstract concept
    Answer: Yes. England is the name of a place. 
    
    Does the phrase or word 'county championship' represent an object, place, individual or title that has a distinct and independant physical existence. Answer no if the word represents a time, date, name of sport or abstract concept
    Answer: No. This word refers to an event which does not have a physical existence. 
    
    Does the phrase or word '[WORD]' represent an object, place, individual or title that has a distinct and independant physical existence. Answer no if the word represents a time, date, name of sport or abstract concept    
    """

    exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
    
    Answer:
    1. bowling | False | as it is an action
    2. Somerset | True | as it is a place
    3. 83 | False | as it is a number
    4. morning | False| as it represents a time of day, with no distinct and independant existence
    5. Grace Road | True | as it is a place or location
    6. Leicestershire | True | as it is the name of a cricket team. 
    7. first innings | False | as it is an abstract concept of a phase in play of cricket
    8. England | True | as it is a place or location
    9. Andy Caddick | True | as it is the name of a person. 
    """
    exemplar_2 = """
    Florian Rousseau ( France ) beat Ainars Kiksis ( Latvia ) 2-0
    
    Answer:
    1. Florian Rousseau | True | as it is the name of a person
    2. France | True | as it is the name of a place or location
    3. beat | False | as it is an action
    4. Ainar Kiksis | True | as it is the name of a person
    5. Latvia | True | as it is the name of a place or location
    6. 2-0 | False | as it is a score or set of numbers which is not an entity. 
    
    """

    def set_config(self, alg, exemplar=False):
        alg.defn = self.defn
        alg.exemplars = [self.exemplar_1, self.exemplar_2]
        if not exemplar:
            alg.phrase_entity_task = self.phrase_entity_task
        else:
            alg.phrase_entity_task = self.phrase_entity_task_exemplar


class GeniaConfig:
    defn = "An entity is a protien, group of protiens, DNA, RNA, Cell Type or Cell Line. " \
           "Abstract concepts, processes and adjectives are not entities"

    phrase_entity_task = "Does the phrase or word '[WORD]' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has " \
                         "a distinct and independant existence. " \
                         "Answer False if the word represents a process, adjective or abstract concept. Explain why"

    def set_config(self, alg, exemplar=False):
        alg.defn = self.defn
        if not exemplar:
            alg.phrase_entity_task = self.phrase_entity_task
        else:
            pass
            #alg.phrase_entity_task = self.phrase_entity_task_exemplar