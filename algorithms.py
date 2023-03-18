import string
import utils
from utils import AnswerMapping
from nltk.corpus import stopwords


class BaseAlgorithm:
    defn = "An entity is an object, place, individual, being, title, proper noun or process that has a distinct and " \
           "independent existence. The name of a collection of entities is also an entity. Adjectives, verbs, numbers, " \
           "adverbs, abstract concepts are not entities. Dates, years and times are not entities"

    # if [] = n then there are O(n^2) phrase groupings

    def __init__(self, model_fn=None, split_phrases=True):
        self.defn = self.defn
        self.para = None
        self.model_fn = model_fn
        self.split_phrases = split_phrases
        self.exemplar_task = None
        self.format_task = None

    def set_para(self, para):
        self.para = para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn


class Algorithm(BaseAlgorithm):
    def perform(self, verbose=True):
        """

        :param model:
        :param paragraph:
        :return:
        """
        answers, metadata = self.perform_single_query(verbose=verbose)
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
        answers = Algorithm.clean_output(answers)
        return answers, metadata

    @staticmethod
    def clean_output(answers):
        answers = list(set(answers))
        for trivial in ["", " ", ".", "-"] + stopwords.words('english'):
            while trivial in answers:
                answers.remove(trivial)
        for i in range(len(answers)):
            ans = answers[i].strip().strip(''.join(string.punctuation)).strip()
            answers[i] = ans
        return answers

    def perform_single_query(self, verbose=True):
        if self.exemplar_task is not None:
            task = self.defn + "\n" + self.exemplar_task + f" '{self.para}' \nAnswer:"
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, verbose=verbose)
        else:
            task = self.defn + "\n" + self.format_task + f"\nParagraph: {self.para} \nAnswer:"
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, verbose=verbose)
        return final, output


class Config:
    cot_format = """
    Format: 
    
    1. First Candidate | True | Explanation why the word is an entity
    2. Second Candidate | False | Explanation why the word is not an entity
    """

    exemplar_format = """
    Format:    
    
    1. First Entity
    2. Second Entity
    """

    def set_config(self, alg, exemplar=True, coT=True, defn=True):
        if defn:
            alg.defn = self.defn
        else:
            alg.defn = ""
        if not exemplar:
            alg.exemplar_task = None
            if coT:
                whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                             "and for each entry explain why it either is or is not an entity. Answer in the format: \n"

                alg.format_task = whole_task + self.cot_format
            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities " \
                             "Answer in the format: \n"

                alg.format_task = whole_task + self.exemplar_format
        else:
            alg.format_task = None
            if coT:
                whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                             "and for each entry explain why it either is or is not an entity. \nParagraph:"
                exemplar_construction = ""
                for exemplar in self.cot_exemplars:
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    exemplar_construction = exemplar_construction + exemplar + "\n"
                exemplar_construction = exemplar_construction + whole_task + "\n"
                alg.exemplar_task = exemplar_construction
            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities \nParagraph:"
                exemplar_construction = ""
                for exemplar in self.exemplars:
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    exemplar_construction = exemplar_construction + exemplar + "\n"
                exemplar_construction = exemplar_construction + whole_task + "\n"
                alg.exemplar_task = exemplar_construction


class ConllConfig(Config):
    defn = "An entity is a person, title, named organization, location, country or nationality." \
           "Names, first names, last names, countries are entities. Nationalities are entities even if they are " \
           "adjectives. Sports, sporting events, adjectives, verbs, numbers, " \
                "adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. " \
           "Possessive words like I, you, him and me are not entities."

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

    cot_exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .
    
    Answer:
    1. money | False | as it is not a named person, organization or location
    2. savings account | False | as it is not a person, organization or location
    3. 5.3 | False | as it is a number
    4. June | False | as it is a date
    5. July | False | as it is a date
    
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

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

    exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

    Answer:
    1. 
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]


class GeniaConfig(Config):
    defn = "An entity is a protien, group of protiens, DNA, RNA, Cell Type or Cell Line. " \
           "Abstract concepts, processes and adjectives are not entities"

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


class CrossNERPoliticsConfig(Config):
    defn = """
    An entity is a person, organization, politician, political party, event, election, country, location or 
    other object that has an independent and distinct existence. Dates, times, abstract concepts, 
    adjectives and verbs are not entities
    """

    cot_exemplar_1 = """
    Assisted by his top aide Harry Hopkins and with very strong national support , 
    he worked closely with British Prime Minister Winston Churchill , Soviet leader Joseph Stalin and 
    Chinese Generalissimo Chiang Kai-shek in leading the Allied Powers against the Axis Powers .
    
    Answer:
    1. Harry Hopkins | True | as it is a person
    2. British | True | as it is a nationality
    3. Prime Minister | False | as it is a title and not a person or politician
    4. Winston Churchill | True | as it is a person
    5. Soviet | True | as it is a nationality
    6. Joseph Stalin | True | as it isa person
    7. Chinese | True | as it is a nationality
    8. Chiang Kai-shek | True | as it is a person
    9. Allied Powers | True | as it is an organization
    10. Axis Powers | True | as it is an organization
    """

    cot_exemplar_2 = """
    Hoover backed conservative leader Robert A. Taft at the 1952 Republican National Convention , 
    but the party 's presidential nomination instead went to Dwight D. Eisenhower , 
    who went on to win the 1952 United States presidential election .

    Answer:
    1. Hoover | True | as it is a person
    2. conservative | False | as it is an ideology and not a political party
    3. Robert A. Taft | True | as it is a person
    4. 1952 Republican National Convention | True | as it is a political event
    5. Dwight D. Eisenhower | True | as it is a person
    6. 1952 United States presidential election | True | as it is an election
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    Assisted by his top aide Harry Hopkins and with very strong national support , 
    he worked closely with British Prime Minister Winston Churchill , Soviet leader Joseph Stalin and 
    Chinese Generalissimo Chiang Kai-shek in leading the Allied Powers against the Axis Powers .

    Answer:
    1. Harry Hopkins
    2. British
    3. Winston Churchill
    4. Soviet
    5. Joseph Stalin
    6. Chinese
    7. Chiang Kai-shek
    8. Allied Powers
    9. Axis Powers
    """

    exemplar_2 = """
    Hoover backed conservative leader Robert A. Taft at the 1952 Republican National Convention , 
    but the party 's presidential nomination instead went to Dwight D. Eisenhower , 
    who went on to win the 1952 United States presidential election .

    Answer:
    1. Hoover
    2. Robert A. Taft
    3. 1952 Republican National Convention
    4. Dwight D. Eisenhower
    5. 1952 United States presidential election
    """

    exemplars = [exemplar_1, exemplar_2]


class CrossNERNaturalSciencesConfig(Config):
    defn = """
    An entity is a person, university, scientist, organization, country, location, scientific discipline, enzyme, 
    protein, chemical compound, chemical element, event, astronomical object, academic journal, award, or theory. 
    Abstract scientific concepts can be entities if they have a name associated with them. 
    Dates, times, adjectives and verbs are not entities
    """

    cot_exemplar_1 = """
    August Kopff , a colleague of Wolf at Heidelberg , then discovered 617 Patroclus eight months after Achilles , 
    and , in early 1907 , he discovered the largest of all Jupiter trojans , 624 Hektor .
    
    Answer:
    1. August Kopff | True | person
    2. Wolf | True | person
    3. Heidelberg | True | as it is a university or location
    4. 617 Patroclus | True | as it is the name of a scientific discovery
    5. Achilles | True | as it is the name of an asteroid
    6. 1907 | False | as it is a date
    7. Jupiter trojans | True | as it is a group of astronomical objects 
    8. 624 Hektor | True | as it is an astronomical object
    """

    cot_exemplar_2 = """
    Nüsslein-Volhard was educated at the University of Tübingen where she earned a PhD in 1974 for research into 
    Protein-DNA interaction s and the binding of RNA polymerase in Escherichia coli .
    
    Answer:
    1. Nüsslein-Volhard | True | as it is a person
    2. University of Tübingen | True | as it is a university
    3. PhD | True | as it is an award
    4. Protein-DNA interaction | True | as it is a scientific discipline
    5. RNA polymerase | True | as it is a chemical compound
    6. Escherichia coli | True | as it is a scientific specimen
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    August Kopff , a colleague of Wolf at Heidelberg , then discovered 617 Patroclus eight months after Achilles , 
    and , in early 1907 , he discovered the largest of all Jupiter trojans , 624 Hektor .

    Answer:
    1. August Kopff
    2. Wolf
    3. Heidelberg
    4. 617 Patroclus
    5. Achilles
    6. Jupiter trojans
    7. 624 Hektor
    """

    exemplar_2 = """
    Nüsslein-Volhard was educated at the University of Tübingen where she earned a PhD in 1974 for research into 
    Protein-DNA interaction s and the binding of RNA polymerase in Escherichia coli .

    Answer:
    1. Nüsslein-Volhard
    2. University of Tübingen
    3. PhD
    4. Protein-DNA interaction
    5. RNA polymerase
    6. Escherichia coli
    """
    exemplars = [exemplar_1, exemplar_2]


class CrossNERMusicConfig(Config):
    defn = """
    An entity is a person, country, location, organization, music genre, song, band, album, artist, musical instrument, 
    award, event or other object with an independent and distinct existence. 
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
    Stevens ' albums Tea for the Tillerman ( 1970 ) and Teaser and the Firecat ( 1971 ) were certified triple platinum 
    in the US by the Recording Industry Association of America .. BBC News .
    
    Answer:
    1. Stevens | True | as it is a name
    2. Tea for Tillerman | True | as it is an album
    3. Teaser and the Firecat | True | as it is an album
    4. 1971 | False | as it is a date
    5. triple platinum | False | as it is an album certification and not an award
    6. US | True | as it is a country
    7. Recording Industry Association of America | True | as it is an organization
    8. BBC News | True | as it is an organization
    """

    cot_exemplar_2 = """
    As a group , the Spice Girls have received a number of notable awards including five Brit Awards , 
    three American Music Awards , three MTV Europe Music Awards , one MTV Video Music Award and three World Music Awards.
    
    Answer:
    1. Spice Girls | True | as it is a band
    2. Brit Awards | True | as it is an award
    3. American Music Awards | True | as it is an award
    4. MTV Europe Music Awards | True | as it is an award
    5. MTV Video Music Award | True | as it is an award
    6. World Music Awards | True | as it is an award
    
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    Stevens ' albums Tea for the Tillerman ( 1970 ) and Teaser and the Firecat ( 1971 ) were certified triple platinum 
    in the US by the Recording Industry Association of America .. BBC News .

    Answer:
    1. Stevens
    2. Tea for Tillerman
    3. Teaser and the Firecat
    6. US
    7. Recording Industry Association of America
    8. BBC News
    """

    exemplar_2 = """
    As a group , the Spice Girls have received a number of notable awards including five Brit Awards , 
    three American Music Awards , three MTV Europe Music Awards , one MTV Video Music Award and three World Music Awards.

    Answer:
    1. Spice Girls
    2. Brit Awards
    3. American Music Awards
    4. MTV Europe Music Awards
    5. MTV Video Music Award
    6. World Music Awards
    """
    exemplars = [exemplar_1, exemplar_2]


class CrossNERLiteratureConfig(Config):
    defn = """
    An entity is a person, country, location, organization, book, writer, poem, magazine, 
    award, event or other object with an independent and distinct existence. 
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
    In 1351 , during the reign of Emperor Toghon Temür of the Yuan dynasty , 93rd-generation descendant Kong Huan 
    ( 孔浣 ) ' s 2nd son Kong Shao ( 孔昭 ) moved from China to Korea during the Goryeo , 
    and was received courteously by Princess Noguk ( the Mongolian-born wife of the future king Gongmin ) .
    
    Answer:
    1. 1351 | False | as it is a date
    2. Emperor Toghon Temür | True | as it is a person
    3. Yuan dynasty | True | as it is the name of a dynasty or organization
    4. Kong Huan | True | as it is the name of a person
    5. 孔浣 | True | as it is a person
    6. Kong Shao | True | as it a person 
    7. 孔昭 | True | as it a person
    8. China | True | as it is a country
    9. Korea | True | as it is a country
    10. Goryeo | True | as it is a event
    11. Princess Noguk | True | as it a person
    12. Mongolian-born | True | as it a nationality
    13. Gongmin | True | as it is a person 
    """

    cot_exemplar_2 = """
    Highly regarded in his lifetime and for a period thereafter , he is now largely remembered for his anti-slavery 
    writings and his poems Barbara Frietchie , The Barefoot Boy , Maud Muller and Snow-Bound .
    
    Answer: 
    1. anti-slavery writings | True | as it is the theme of writing of some works
    2. poems | True | as it is the word poem
    3. Barbara Frietchie | True | as it is a poem
    4. The Barefoot Boy | True | as it is a poem
    5. Maud Muller | True | as it is a poem
    6. Snow-Bound | True | as it is a poem 
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    In 1351 , during the reign of Emperor Toghon Temür of the Yuan dynasty , 93rd-generation descendant Kong Huan 
    ( 孔浣 ) ' s 2nd son Kong Shao ( 孔昭 ) moved from China to Korea during the Goryeo , 
    and was received courteously by Princess Noguk ( the Mongolian-born wife of the future king Gongmin ) .

    Answer:
    1. Emperor Toghon Temür
    2. Yuan dynasty
    3. Kong Huan
    4. 孔浣
    5. Kong Shao
    6. 孔昭
    7. China
    8. Korea
    9. Goryeo
    10. Princess Noguk
    11. Mongolian-born
    12. Gongmin
    """

    exemplar_2 = """
    Highly regarded in his lifetime and for a period thereafter , he is now largely remembered for his anti-slavery 
    writings and his poems Barbara Frietchie , The Barefoot Boy , Maud Muller and Snow-Bound .

    Answer: 
    1. anti-slavery writings
    2. poems
    3. Barbara Frietchie
    4. The Barefoot Boy
    5. Maud Muller
    6. Snow-Bound
    """

    exemplars = [exemplar_1, exemplar_2]


class CrossNERAIConfig(Config):
    defn = """
    An entity is a person, country, location, organization, field of Artificial Intelligence, 
    task in artificial intelligence, product, algorithm, metric in artificial intelligence, university. 
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
    Popular approaches of opinion-based recommender system utilize various techniques including text mining , 
    information retrieval , sentiment analysis ( see also Multimodal sentiment analysis ) and deep learning X.Y. Feng , 
    H. Zhang , 21 ( 5 ) : e12957 .
    
    Answers:
    1. opinion-based recommender system | True | as it is a type of system in AI
    2. text mining | True | as it is a technique or method in AI
    3. information retrieval | True | as it is a technique or method in AI
    4. sentiment analysis | True | as it is a technique or method in AI
    5. Multimodal sentiment analysis | True | as it is a technique or method in AI
    6. deep learning | True | as it is a technique or method in AI
    7. X.Y. Feng | True | as it is a person
    8. H. Zhang | True | as it is a person
    """

    cot_exemplar_2 = """
    Octave helps in solving linear and nonlinear problems numerically , and for performing other numerical experiments 
    using a that is mostly compatible with MATLAB.
    
    Answers:
    1. Octave | True | as it is a product or tool
    2. linear and nonlinear problems | False | as it is a type of problem
    3. MATLAB | True | as it is a product or tool
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    exemplar_1 = """
    Popular approaches of opinion-based recommender system utilize various techniques including text mining , 
    information retrieval , sentiment analysis ( see also Multimodal sentiment analysis ) and deep learning X.Y. Feng , 
    H. Zhang , 21 ( 5 ) : e12957 .

    Answers:
    1. opinion-based recommender system
    2. text mining
    3. information retrieval
    4. sentiment analysis
    5. Multimodal sentiment analysis
    6. deep learning
    7. X.Y. Feng
    8. H. Zhang
    """

    exemplar_2 = """
    Octave helps in solving linear and nonlinear problems numerically , and for performing other numerical experiments 
    using a that is mostly compatible with MATLAB.

    Answers:
    1. Octave
    2. MATLAB
    """
    exemplars = [exemplar_1, exemplar_2]


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