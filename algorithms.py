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

    def __init__(self, model_fn=None, word_limit=4, split_phrases=True, mode=1):
        self.defn = self.defn.replace("[WORDLIMIT]", f"{word_limit}")
        self.para = None
        self.model_fn = model_fn
        self.mode = mode
        self.split_phrases = split_phrases
        self.exemplars = None

    def set_para(self, para):
        self.para = para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    def initial_query(self, verbose=False, indent_level=0):
        if self.exemplars is not None:
            q = self.defn + "\n"
            for exemplar in self.exemplars:
                q += self.base_task + "\nParagraph: " + exemplar + "\n"
            q += "\n" + self.defn + "\n" + self.base_task + "\nParagraph: " + self.para + "\nAnswer:"
        else:
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


class CapitalAlgorithm(BaseAlgorithm):
    def is_date(self, word):
        return word.lower() in ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                "october", "november", "december"]

    def perform(self, verbose=False):
        words = self.para.split()
        selected = []
        for i, word in enumerate(words):
            if i == 0:
                is_entity = self.check_word(word, verbose=verbose, indent_level=1)
                if is_entity:
                    selected.append(word)
            else:
                if len(word) <= 1:
                    pass
                else:
                    if word[0].isupper():
                        if not self.is_date(word):
                            selected.append(word)
        return selected, None


class Algorithm(BaseAlgorithm):
    def perform(self, verbose=True):
        """

        :param model:
        :param paragraph:
        :param mode: 0: most exhaustive, 2: least exhaustive
        :return:
        """
        if self.mode == 0:
            answers, metadata = self.perform_exhaustive(verbose=verbose)
        elif self.mode == 1:
            answers, metadata = self.perform_single_query(verbose=verbose)
        else:
            answers = None
        answers = list(set(answers))
        if self.split_phrases:
            new_answers = []
            for answer in answers:
                if " " not in answer:
                    new_answers.append(answer)
                else:
                    minis = answer.split(" ")
                    for mini in minis:
                        new_answers.append(mini)
            answers = new_answers
        answers = list(set(answers))
        for trivial in ["", " ", "."] + stopwords.words('english'):
            while trivial in answers:
                answers.remove(trivial)
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


class Config:
    generic_cot_exemplar_1 = """
    It's a skateboarding penguin with a sunhat!

    Answer:
    1. It's | False | as it is a pronoun
    2. Skateboarding | False | as it is an action or adjective, with no distinct existence 
    3. Penguin | True | as it is an animal with a distinct and independent existence. 
    4. Sunhat | True | as it is an object with a distinct and independent existence. 
    """
    generic_cot_exemplar_2 = """
    Smith saw boulders lined the side of the road, foretelling what could come next.

    Answer:
    1. Smith | True | as it is the name of a person
    2. boulders | True | as they are objects with independent and distinct existence. 
    3. side | False | as it is a position with no distinct existence
    4. road | True | as it is an object with a distinct and independent existence. 
    5. foretelling | False | as it is an action

    """
    generic_cot_exemplars = [generic_cot_exemplar_1, generic_cot_exemplar_2]

    generic_exemplar_1 = """
    It's a skateboarding penguin with a sunhat!

    Answer:
    1. Penguin
    2. Sunhat
    """
    generic_exemplar_2 = """
    Smith saw boulders lined the side of the road, foretelling what could come next.

    Answer:
    1. Smith
    2. boulders
    3. road
    """
    generic_exemplars = [generic_exemplar_1, generic_exemplar_2]

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
        """

    def set_config(self, alg, exemplar=True, coT=True, generic=False):
        alg.defn = self.defn
        if not exemplar:
            alg.phrase_entity_task = self.phrase_entity_task
        else:
            if not generic:
                if coT:
                    alg.exemplars = self.cot_exemplars
                else:
                    alg.exemplars = self.exemplars
            else:
                if coT:
                    alg.exemplars = self.generic_cot_exemplars
                else:
                    alg.exemplars = self.generic_exemplars
            alg.phrase_entity_task = self.phrase_entity_task_exemplar


class ConllConfig(Config):
    defn = "An entity is an object, place, individual, being, title or process that has a distinct and " \
                "independent existence. The name of a collection of entities is also an entity. " \
           "Names, first names, last names, countries are entities. Nationalities are entities even if they are " \
           "adjectives. Sports, sporting events, adjectives, verbs, numbers, " \
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
    Answer: 
    """

    cot_exemplar_1 = """
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
    cot_exemplar_2 = """
    Florian Rousseau ( France ) beat Ainars Kiksis ( Latvia ) 2-0
    
    Answer:
    1. Florian Rousseau | True | as it is the name of a person
    2. France | True | as it is the name of a place or location
    3. beat | False | as it is an action
    4. Ainar Kiksis | True | as it is the name of a person
    5. Latvia | True | as it is the name of a place or location
    6. 2-0 | False | as it is a score or set of numbers which is not an entity. 
    
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Answer:
    1. Somerset
    2. Grace Road
    3. Leicestershire
    4. England
    5. Andy Caddick
    """
    exemplar_2 = """
    Florian Rousseau ( France ) beat Ainars Kiksis ( Latvia ) 2-0

    Answer:
    1. Florian Rousseau
    2. France
    3. Ainar Kiksis
    4. Latvia
    """
    exemplars = [exemplar_1, exemplar_2]


class GeniaConfig(Config):
    defn = "An entity is a protien, group of protiens, DNA, RNA, Cell Type or Cell Line. " \
           "Abstract concepts, processes and adjectives are not entities"

    phrase_entity_task = "Does the phrase or word '[WORD]' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has " \
                         "a distinct and independant existence. " \
                         "Answer False if the word represents a process, adjective or abstract concept. Explain why"



    phrase_entity_task_exemplar = """
        Does the phrase or word 'cytokines' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has a distinct and independant existence. Answer False if the word represents a process, adjective or abstract concept. Explain why
        Answer: Yes. Cytokines is a type of protein that is made by certain immune and non-immune cells and has an effect on the immune system. 

        Does the phrase or word 'AP-1' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has a distinct and independant existence. Answer False if the word represents a process, adjective or abstract concept. Explain why
        Answer: Yes. Activator protein 1 (AP-1) is a transcription factor DNA that regulates gene expression in response to a variety of stimuli

        Does the phrase or word 'anti-CD4 mAb' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has a distinct and independant existence. Answer False if the word represents a process, adjective or abstract concept. Explain why
        Answer: Yes. anti-CD4 mAb is a cell type

        Does the phrase or word 'CD4+ T cells' represent a protien, group of protiens, DNA, RNA, Cell Type or Cell Line that has a distinct and independant existence. Answer False if the word represents a process, adjective or abstract concept. Explain why
        Answer: Yes. This is because CD4+ T Cells is a Cell Type

        """

    cot_exemplar_1 = """
        Immunoprecipitation of the gp 160 -induced nuclear extracts with polyclonal antibodies to Fos and Jun proteins indicates that AP-1 complex is comprised of members of these family of proteins.

        Answer:
        1. Immunoprecipitation | False | as it is a process
        2. gp 160  | True | as Glycoprotein Gp 160 it is a type of protein
        3. polyclonal antibodies | True | as it is a type of cell
        4. Fos | True | as Fructo-oligosaccharides are proteins
        5. Jun | True | as it is a type of protein
        6. AP-1 | True | as Activator protein 1 (AP-1) is a protein
        """
    cot_exemplar_2 = """
        The stimulatory effects of gp160 are mediated through the CD4 molecule , since treatment of gp160 with soluble CD4-IgG abrogates its activity , and CD4 negative T cell lines fail to be stimulated with gp160 

        Answer:
        1. gp 160 | True | as Glycoprotein Gp 160 it is a type of protein
        2. mediated | False | as it is a verb
        3. CD4 molecule | True | as CD4 (cluster of differentiation 4) is a glycoprotein a type of protien
        4. CD4-IgG | True | as CD4-igG is a homodimer of a hybrid polypeptide
        5. abrogates | False | as it is a verb
        6. CD4 negative T cell lines | True | as they are a type of Cell Line

        """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
        Immunoprecipitation of the gp 160 -induced nuclear extracts with polyclonal antibodies to Fos and Jun proteins indicates that AP-1 complex is comprised of members of these family of proteins.

        Answer:
        1. gp 160
        2. polyclonal antibodies
        3. Fos
        4. Jun
        5. AP-1
        """
    exemplar_2 = """
        The stimulatory effects of gp160 are mediated through the CD4 molecule , since treatment of gp160 with soluble CD4-IgG abrogates its activity , and CD4 negative T cell lines fail to be stimulated with gp160 

        Answer:
        1. gp 160
        2. CD4 molecule
        3. CD4-IgG
        4. CD4 negative T cell lines
        """
    exemplars = [exemplar_1, exemplar_2]


class CrossNERConfig(Config):
    # No defn or phrase_entity_task
    cot_exemplar_1 = ConllConfig.cot_exemplar_1
    cot_exemplar_2 = """
    India carried out a nuclear test in 1974 but says it has not built the bomb .
    
    Answer:
    1. India | True | as it is a country
    2. nuclear test | False | as it is a concept and not an entity with physical existence
    3. 1974 | False | as it is a date or time
    4. bomb | True | as it is an object with independent and distinct existence. 
    
    """
    exemplar_1 = ConllConfig.exemplar_1
    exemplar_2 = """
    India carried out a nuclear test in 1974 but says it has not built the bomb .
    
    Answer:
    1. India
    2. bomb
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]
    exemplars = [exemplar_1, exemplar_2]


class CrossNERPoliticsConfig(CrossNERConfig):
    defn = """
    An entity is a person, organization, politician, political party, event, eleection, country, location or 
    other object that has an independent and distinct existence. Dates, times, abstract concepts, 
    adjectives and verbs are not entities
    """

    phrase_entity_task = """
    Does the phrase or word '[WORD]' represent a person, organization, politician, political party, event, eleection, country, location or 
    other object that has an independent and distinct existence? Answer False if the word represents 
    dates, times, abstract concepts, adjectives and verbs as these are not entities. Explain why
    """


class CrossNERNaturalSciencesConfig(CrossNERConfig):
    defn = """
    An entity is a person, university, scientist, organization, country, location, scientific discipline, enzyme, 
    protein, chemical compound, chemical element, event, astronomical object, academic journal, award, theory or 
    other object that has an independent and distinct existence. Abstract scientific concepts can be entities if they 
    have a name associated with them. Dates, times, adjectives and verbs are not entities
    """

    phrase_entity_task = """
    Does the phrase or word '[WORD]' represent a person, university, scientist, organization, country, location, 
    scientific discipline, enzyme, 
    protein, chemical compound, chemical element, event, astronomical object, academic journal, award, theory or 
    other object that has an independent and distinct existence? Answer False if the word represents 
    dates, times, adjectives and verbs as these are not entities. Explain why
    """


class CrossNERMusicConfig(CrossNERConfig):
    defn = """
    An entity is a person, country, location, organization, music genre, song, band, album, artist, musical instrument, 
    award, event or other object with an independent and distinct existence. 
    Dates, times, adjectives and verbs are not entities. 
    """

    phrase_entity_task = """
        Does the phrase or word '[WORD]' represent a person, country, location, organization, music genre, song, band, 
        album, artist, musical instrument, 
        award, event or other object with an independent and distinct existence? Answer False if the word represents 
        dates, times, adjectives and verbs as these are not entities. Explain why
    """


class CrossNERLiteratureConfig(CrossNERConfig):
    defn = """
    An entity is a person, country, location, organization, book, writer, poem, magazine, 
    award, event or other object with an independent and distinct existence. 
    Dates, times, adjectives and verbs are not entities. 
    """

    phrase_entity_task = """
        Does the phrase or word '[WORD]' represent a person, country, location, organization, book, writer, poem, 
        magazine,
        award, event or other object with an independent and distinct existence? Answer False if the word represents 
        dates, times, adjectives and verbs as these are not entities. Explain why
    """


class CrossNERAIConfig(CrossNERConfig):
    defn = """
    An entity is a person, country, location, organization, field of Artificial Intelligence, 
    task in artificial intelligence, product, algorithm, metric in artificial intelligence, university. 
    Dates, times, adjectives and verbs are not entities. 
    """

    phrase_entity_task = """
        Does the phrase or word '[WORD]' represent a person, country, location, organization, 
        field of Artificial Intelligence, task in artificial intelligence, product, algorithm, 
        metric in artificial intelligence, university
        award, event or other object with an independent and distinct existence? Answer False if the word represents 
        dates, times, adjectives and verbs as these are not entities. Explain why
    """


class FewNERDConfig(Config):
    person = "person"
    art = "piece of art"
    miscellaneous = "product, language, living thing, currency, god or scientific concept in astronomy, biology etc. "
    locations = "locations"
    organizations = "organizations"
    buildings = "the names of buildings"
    events = "events"
    clearly_not = "Dates, times, abstract concepts and adjectives"
    train_group = f"{person}, {art}, {miscellaneous}."
    dev_group = f"{buildings} and {events}"
    test_group = f"{locations} and {organizations}"
    q_1 = "Albert Einstein used 100 USD to purchase the Eiffel tower from the Association of Artificial Intelligence"
    q_2 = "In England, there is a festival called the Grand Jubilee, founded in 1982 by Attila the Hun, " \
          "it was the original birthplace of the painting 'The Starry Night'"


class FewNERDINTRATrainConfig(FewNERDConfig):
    defn = f"""
    An entitiy is a {FewNERDConfig.train_group}. {FewNERDConfig.dev_group} are not entities, 
    {FewNERDConfig.test_group} are also not entities. {FewNERDConfig.clearly_not} are not entities. 
    """
    cot_exemplar_1 = FewNERDConfig.q_1 + \
        """
         Answer:
         1. Albert Einstein | True | as this is the name of a person
         2. USD | True | as this is the name of a currency
         3. purchase | False | as this is an action or verb
         4. Eiffel tower | False | as this is the name of a building
         5. Association of Artificial Intelligence | False | as this is an organization
        """

    cot_exemplar_2 = FewNERDConfig.q_2 + \
        """
        Answer:
        1. England | False | as it is a location
        2. festival | False | as it is not a named entity
        3. Grand Jubilee | False | as it is an event
        4. 1982 | False | as it is a date
        5. Attila the Hun | True | as it is a person
        6. The Starry Night | True | as it is a piece of art
        """

    exemplar_1 = FewNERDConfig.q_1 + \
        """
        Answer:
        1. Albert Einstein
        2. USD
        """

    exemplar_2 = FewNERDConfig.q_2 + \
        """
        Answer: 
        1. Attila the Hun
        2. The Starry Night
        """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]
    exemplars = [exemplar_1, exemplar_2]


class FewNERDINTRADevConfig(FewNERDConfig):
    defn = f"""
        Entities are  {FewNERDConfig.dev_group}. Entities are not a {FewNERDConfig.train_group}, 
        {FewNERDConfig.test_group} are also not entities. {FewNERDConfig.clearly_not} are not entities. 
        """

    cot_exemplar_1 = FewNERDConfig.q_1 + \
         """
          Answer:
          1. Albert Einstein | False | as this is the name of a person
          2. USD | False | as this is the name of a currency
          3. purchase | False | as this is an action or verb
          4. Eiffel tower | True | as this is the name of a building
          5. Association of Artificial Intelligence | False | as this is an organization
         """

    cot_exemplar_2 = FewNERDConfig.q_2 + \
         """
         Answer:
         1. England | False | as it is a location
         2. festival | False | as it is not a named entity
         3. Grand Jubilee | True | as it is an event
         4. 1982 | False | as it is a date
         5. Attila the Hun | False | as it is a person
         6. The Starry Night | False | as it is a piece of art
         """

    exemplar_1 = FewNERDConfig.q_1 + \
         """
         Answer:
         1. Eiffel Tower
         """

    exemplar_2 = FewNERDConfig.q_2 + \
         """
         Answer: 
         1. Grand Jubilee
         """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]
    exemplars = [exemplar_1, exemplar_2]


class FewNERDINTRATestConfig(FewNERDConfig):
    defn = f"""
        Entities are  {FewNERDConfig.test_group}. Entities are not a {FewNERDConfig.train_group}, 
        {FewNERDConfig.dev_group} are also not entities. {FewNERDConfig.clearly_not} are not entities. 
        """

    cot_exemplar_1 = FewNERDConfig.q_1 + \
         """
          Answer:
          1. Albert Einstein | False | as this is the name of a person
          2. USD | False | as this is the name of a currency
          3. purchase | False | as this is an action or verb
          4. Eiffel tower | False | as this is the name of a building
          5. Association of Artificial Intelligence | True | as this is an organization
         """

    cot_exemplar_2 = FewNERDConfig.q_2 + \
         """
         Answer:
         1. England | True | as it is a location
         2. festival | False | as it is not a named entity
         3. Grand Jubilee | False | as it is an event
         4. 1982 | False | as it is a date
         5. Attila the Hun | False | as it is a person
         6. The Starry Night | False | as it is a piece of art
         """

    exemplar_1 = FewNERDConfig.q_1 + \
         """
         Answer:
         1. Association of Artificial Intelligence
         """

    exemplar_2 = FewNERDConfig.q_2 + \
         """
         Answer: 
         1. England
         """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]
    exemplars = [exemplar_1, exemplar_2]