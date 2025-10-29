
<div align="center">
  <img src="https://github.com/SapienzaNLP/xcore/blob/master/media/xcore_logo.png" height="350">
</div>
<h1 align="center">
  All-in-one model for end-to-end Coreference Resolution
</h1>



##  Description
This repository contains the official code for the EMNLP 2025 main conference paper:  [*xCoRe:
Cross-Context Coreference Resolution*]() by [Giuliano Martinelli](https://www.linkedin.com/in/giuliano-martinelli-20a9b2193/), [Bruno Gatti](https://www.linkedin.com/in/tommaso-bonomo/) and [Roberto Navigli](https://www.linkedin.com/in/robertonavigli/).

xCoRe is the first all-in-one model that has best-in-class performance for every coreference resolution setting, working seamlessly with short, medium-sized, long, and multiple documents.

xCoRe models leverage the techniques used in [Maverick](https://pypi.org/project/maverick-coref/) for within-context coreference resolution and for medium-sized inputs.
When dealing with long documents or multiple inputs, xCoRe first computes within-context clusters and then uses a supervised cross-context cluster merging technique to compute final coreference clusters.

# Setup
Install the library from [PyPI](https://pypi.org/project/xcore-coref/)

```bash
pip install xcore-coref
```
or from source 

```bash
git clone https://github.com/SapienzaNLP/xcore.git
cd xcore
pip install -e .
```

## Loading a Pretrained Model
xCoRe models can be loaded using huggingface_id or local path:
```bash
from xcore-coref import xcore
model = xcore(
  hf_name_or_path = "model_name" | "model_path", default = "sapienzanlp/xcore-litbank"
  device = "cpu" | "cuda", default = "cuda:0"
)
```

## Available Models

Models are available at [SapienzaNLP huggingface hub](https://huggingface.co/collections/sapienzanlp/xcore-models):

|            hf_model_name            | dataset | Score | Mode |
|:-----------------------------------:|:----------------:|:-----:|:----------:|
|    ["sapienzanlp/xcore-litbank"](https://huggingface.co/sapienzanlp/xcore-litbank)    |     [LitBank](https://aclanthology.org/2020.lrec-1.6/)    |  78.3 |    Single Document  (Book Splits)|
|      ["sapienzanlp/xcore-ecb"](https://huggingface.co/sapienzanlp/xcore-ecb)      |       [ECB+](https://aclanthology.org/L14-1646/)      |  42.4 |     Multiple Documents  (News)  |
|      ["sapienzanlp/xcore-scico"](https://huggingface.co/sapienzanlp/xcore-scico)      |       [SciCo](https://arxiv.org/abs/2104.08809)      |  31.0 |     Multiple Documents (Scientific)    |
<!--|    ["sapienzanlp/xcore-all"](https://huggingface.co/sapienzanlp/xcore-all)    |     LitBank, ECB+, BookCoref_silver, SciCo, OntoNotes, PreCo   |  - |   All datasets |-->
<!--|     ["sapienzanlp/xcore-bookcoref"](https://huggingface.co/sapienzanlp/xcore-bookcoref)     |      [BookCoref](https://huggingface.co/datasets/sapienzanlp/bookcoref)     |  62.3 |     Single Document  (Full Books)  |-->
N.B. Each dataset has different annotation guidelines; choose your model according to your use case, as explained in the paper.

## Inference
### Input Types
Inputs can be formatted as either single or multiple documents.
When single documents are very long, they are split into multiple subdocuments and coreference is performed in a cross-context fashion.

Input documents can be either:
- plain text:
  ```bash
  text = "Barack Obama is traveling to Rome. The city is sunny and the president plans to visit its most important attractions"
  ```
- word-tokenized text, as a list of tokens:
  ```bash
  word_tokenized = ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.',  'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']
  ```
- sentence-split, word-tokenized text, i.e., OntoNotes-like input, as a list of lists of tokens:
  ```bash
  ontonotes_format = [['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.'], ['The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']] 
  ```


### Predict
You can use: 
```
model.predict()
````
to obtain coreference predictions of any kind of input text.
The model will return a dictionary containing:
- `tokens`, word tokenized version of the input.
- `clusters_token_offsets`, a list of clusters containing mentions' token offsets as a tuple (start, end). When dealing with multiple documents, it becomes a triplet (doc, start, end) in which doc is the index of the document in input.
- `clusters_text_mentions`, a list of clusters containing mentions in plain text.

#### Examples
**Traditional short-/ medium-sized inputs:**
For standard inputs that do not exceed thousands of tokens, it is just necessary to input the model as it is.
The model will only perform within-window coreference as in Maverick, and give back the list of coreference resolution predictions.
  ```bash
short_text = [['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.'], ['The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']] 
model.predict(short_text)
>>> {
  'tokens': ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.', 'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'monument', ',', 'the', 'Colosseum'], 
  'clusters_token_offsets': [[(5, 5), (7, 8), (17, 17)], [(0, 1), (12, 13)]],  # (start, end)
  'clusters_text_mentions': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]
}
```

**Long-document Coreference:**
For very long inputs, such as full narrative books or very long scientific articles, the model requires two parameters:
1. specify "long" as `input_type`,
2. insert the `max_length` parameter.
Max_length only depends on your current device and memory space, and we suggest inputting the maximum length possible to have the highest performance and efficiency at inference time.
  ```bash
from datasets import load_dataset
full_book = load_dataset("sapienzanlp/bookcoref")["train"][0]

model.predict(full_book, "long", max_length=4000)
>>> {
  'tokens': ['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.', 'The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'monument', ',', 'the', 'Colosseum'], 
  'clusters_token_offsets': [[(5, 5), (7, 8), (17, 17)], [(0, 1), (12, 13)]], # (start, end)
  'clusters_text_mentions': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]
}
```

**Cross-document Coreference:**
For multiple documents, the model will encode them into separate windows and perform end-to-end cross-document coreference
```bash
text1 = [['Barack', 'Obama', 'is', 'traveling', 'to', 'Rome', '.'], ['The', 'city', 'is', 'sunny', 'and', 'the', 'president', 'plans', 'to', 'visit', 'its', 'most', 'important', 'attractions']] 
text2 = [...]
text3 = [...]
model.predict([text1, text2, text3], "cross")
>>> {
  'tokens': [[text1_tokens],[text1_tokens],[text1_tokens]], 
  'clusters_token_offsets': [[(0, 5, 5), (0, 7, 8), (2, 17, 17)], [(1, 0, 1), (2, 12, 13)]], # (document_offset, start, end)
  'clusters_text_mentions': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president']]
  }
```


**Singletons:** 
For all of the above settings, either include or exclude singletons (i.e., single mention clusters) prediction by setting `singletons` to `True` or `False`.
*(hint: for accurate singletons use PreCo- or LitBank-based models, since OntoNotes does not include singletons and therefore the model is not trained to extract any)*
  ```bash
  model.predict(ontonotes_format, singletons=True)
  {'tokens': [...], 
  'clusters_token_offsets': [((5, 5), (7, 8), (17, 17)), ((0, 1), (12, 13)), ((17, 20),)],
  'clusters_char_offsets': None, 
  'clusters_token_text': [['Rome', 'The city', 'its'], ['Barack Obama', 'the president'], ['its most important attractions']], 
  'clusters_char_text': None
  }
  ```

# Using the Official Training and Evaluation Script

This same repository also contains the code to train and evaluate xCoRe systems using 'pytorch-lightning' and 'Hydra'.

**We strongly suggest to directly use the [python package](https://pypi.org/project/xcore-coref/) for easier inference and downstream usage.** 

## Environment
To set up the training and evaluation environment, run the bash script setup.sh that you can find at the top level in this repository. This script will handle the creation of a new conda environment and will take care of all the requirements and data preprocessing for training and evaluating a model on OntoNotes. 

Simply run the following command:
```
bash ./setup.sh
```

## Data 
Official Links:
- [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)
- [PreCo](https://drive.google.com/file/d/1q0oMt1Ynitsww9GkuhuwNZNq6SjByu-Y/view)
- [LitBank](https://github.com/dbamman/litbank/tree/master/coref/conll)
- [BookCoref](https://huggingface.co/datasets/sapienzanlp/bookcoref)
- [ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus/)
- [SciCo](https://huggingface.co/datasets/allenai/scico)

Since those datasets usually require a preprocessing step to obtain the OntoNotes-like jsonlines format, except for BookCoref, which we suggest to download with [huggingface](https://huggingface.co/datasets/sapienzanlp/bookcoref), we release a ready-to-use version:
https://drive.google.com/drive/u/3/folders/18dtd1Qt4h7vezlm2G0hF72aqFcAEFCUo. 


## Hydra
This repository uses the [Hydra](https://hydra.cc/) configuration environment.

- In *conf/data/* each yaml file contains a dataset configuration.
- *conf/evaluation/* contains the model checkpoint file path and device settings for model evaluation.
- *conf/logging/* contains details for wandb logging.
- In *conf/model/*, each yaml file contains a model setup.
-  *conf/train/* contains training configurations.
- *conf/root.yaml* regulates the overall configuration of the environment.


## Train
To train an xCoRe model, modify *conf/root.yaml* with your custom setup. 
By default, this file contains the settings for training and evaluating on the OntoNotes dataset.

To train a new model, follow the steps in the [Environment](#environment) section and run the following script:
```
conda activate xcore_env
python xcore/train.py
```

## Evaluate
To evaluate an existing model, it is necessary to set up two different environment variables.
1. Set the dataset path in `conf/root.yaml`.
2. Set the model checkpoint path in `conf/evaluation/default_evaluation.yaml`.

Finally, run the following:
```
conda activate xcore_env
python xcore/evaluate.py
```
This will directly output the CoNLL-2012 scores, and, under the experiments/ folder, an output.jsonlines file containing the model outputs in OntoNotes style.


# Citation
This work has been published at the [EMNLP 2025 main conference](https://aclanthology.org). 
If you use any part, please consider citing our paper as follows:
```bibtex
@inproceedings{martinelli-etal-2025-xcore,
    title = "xCoRe: Cross-context Coreference Resolution",
    author = "Martinelli, Giuliano  and
      Gatti, Bruno and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, Cina",
    publisher = "Association for Computational Linguistics",

```

## License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).


