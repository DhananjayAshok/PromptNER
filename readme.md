# PromptNER
## Prompting For Named Entity Recognition

This is the home of the PromptNER tool, currently anonymized for conference submission purposes. 

## Installation
Very little is needed by way of installation, simply clone and install the requirements (we use Python 3.10.9 but many other versions should work without a problem)

You will also need to download nltk stopwords:
```sh
git clone https://github.com/GitGudAtNLP/PromptNER PromptNER 
cd PromptNER
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

You will have to set up an [OpenAI account](https://platform.openai.com/overview) to use the GPT based Prompting models. Make sure to save the API key as an environment variable: (put this line in your ~/.bashrc file if you are on Ubuntu to avoid doing this everytime)

```sh
export OPENAI_API_KEY='YOUR API KEY HERE'
```

## Running PromptNER on your own dataset or task

All PromptNER needs to perform Named Entity Recognition on a completely unseen task specification is
- A definition of the concept of a named entity, written in natural language
- A few examples (3-5 is usually good) of the task being done succesfully
- 
Head on to the [custom]() file and follow the ExampleConfig class to add in your desired specifications in the CustomClass, after this you should be able to import the 'get_ner_system' function and query it to get a function which takes in a sentence in text and outputs a list of extracted entities. 

## Reproducing Experiments
To reproduce the experiments in the paper visit the [run]() file, you can adapt the main function to call either 'run_all_datasets' or 'ablate_best' to get all the results from the paper. 

You will have to download the data from the [drive link]() and extract it at root level of the repository. 