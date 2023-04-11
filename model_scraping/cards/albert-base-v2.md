# ALBERT Base v2

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://arxiv.org/abs/1909.11942) and first released in
[this repository](https://github.com/google-research/albert). This model, as all ALBERT models, is uncased: it does not make a difference
between english and English.

Disclaimer: The team releasing ALBERT did not write a model card for this model so this model card has been written by
the Hugging Face team.

## Model description

ALBERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it
was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Sentence Ordering Prediction (SOP): ALBERT uses a pretraining loss based on predicting the ordering of two consecutive segments of text.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the ALBERT model as inputs.

ALBERT is particular in that it shares its layers across its Transformer. Therefore, all layers have the same weights. Using repeating layers results in a small memory footprint, however, the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

This is the second version of the base model. Version 2 is different from version 1 due to different dropout rates, additional training data, and longer training. It has better results in nearly all downstream tasks.

This model has the following configuration:

- 12 repeating layers
- 128 embedding dimension
- 768 hidden dimension
- 12 attention heads
- 11M parameters

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=albert) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='albert-base-v2')
>>> unmasker("Hello I'm a [MASK] model.")
[
   {
      "sequence":"[CLS] hello i'm a modeling model.[SEP]",
      "score":0.05816134437918663,
      "token":12807,
      "token_str":"▁modeling"
   },
   {
      "sequence":"[CLS] hello i'm a modelling model.[SEP]",
      "score":0.03748830780386925,
      "token":23089,
      "token_str":"▁modelling"
   },
   {
      "sequence":"[CLS] hello i'm a model model.[SEP]",
      "score":0.033725276589393616,
      "token":1061,
      "token_str":"▁model"
   },
   {
      "sequence":"[CLS] hello i'm a runway model.[SEP]",
      "score":0.017313428223133087,
      "token":8014,
      "token_str":"▁runway"
   },
   {
      "sequence":"[CLS] hello i'm a lingerie model.[SEP]",
      "score":0.014405295252799988,
      "token":29104,
      "token_str":"▁lingerie"
   }
]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained("albert-base-v2")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import AlbertTokenizer, TFAlbertModel
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2'')
model = TFAlbertModel.from_pretrained("albert-base-v2)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

### Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased
predictions:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='albert-base-v2')
>>> unmasker("The man worked as a [MASK].")

[
   {
      "sequence":"[CLS] the man worked as a chauffeur.[SEP]",
      "score":0.029577180743217468,
      "token":28744,
      "token_str":"▁chauffeur"
   },
   {
      "sequence":"[CLS] the man worked as a janitor.[SEP]",
      "score":0.028865724802017212,
      "token":29477,
      "token_str":"▁janitor"
   },
   {
      "sequence":"[CLS] the man worked as a shoemaker.[SEP]",
      "score":0.02581118606030941,
      "token":29024,
      "token_str":"▁shoemaker"
   },
   {
      "sequence":"[CLS] the man worked as a blacksmith.[SEP]",
      "score":0.01849772222340107,
      "token":21238,
      "token_str":"▁blacksmith"
   },
   {
      "sequence":"[CLS] the man worked as a lawyer.[SEP]",
      "score":0.01820771023631096,
      "token":3672,
      "token_str":"▁lawyer"
   }
]

>>> unmasker("The woman worked as a [MASK].")

[
   {
      "sequence":"[CLS] the woman worked as a receptionist.[SEP]",
      "score":0.04604868218302727,
      "token":25331,
      "token_str":"▁receptionist"
   },
   {
      "sequence":"[CLS] the woman worked as a janitor.[SEP]",
      "score":0.028220869600772858,
      "token":29477,
      "token_str":"▁janitor"
   },
   {
      "sequence":"[CLS] the woman worked as a paramedic.[SEP]",
      "score":0.0261906236410141,
      "token":23386,
      "token_str":"▁paramedic"
   },
   {
      "sequence":"[CLS] the woman worked as a chauffeur.[SEP]",
      "score":0.024797942489385605,
      "token":28744,
      "token_str":"▁chauffeur"
   },
   {
      "sequence":"[CLS] the woman worked as a waitress.[SEP]",
      "score":0.024124596267938614,
      "token":13678,
      "token_str":"▁waitress"
   }
]
```

This bias will also affect all fine-tuned versions of this model.

## Training data

The ALBERT model was pretrained on [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038
unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and
headers).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using SentencePiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

### Training

The ALBERT procedure follows the BERT setup.

The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

## Evaluation results

When fine-tuned on downstream tasks, the ALBERT models achieve the following results:

|                | Average  | SQuAD1.1 | SQuAD2.0 | MNLI     | SST-2    | RACE     |
|----------------|----------|----------|----------|----------|----------|----------|
|V2              |
|ALBERT-base     |82.3      |90.2/83.2 |82.1/79.3 |84.6      |92.9      |66.8      |
|ALBERT-large    |85.7      |91.8/85.2 |84.9/81.8 |86.5      |94.9      |75.2      |
|ALBERT-xlarge   |87.9      |92.9/86.4 |87.9/84.1 |87.9      |95.4      |80.7      |
|ALBERT-xxlarge  |90.9      |94.6/89.1 |89.8/86.9 |90.6      |96.8      |86.8      |
|V1              |
|ALBERT-base     |80.1      |89.3/82.3 | 80.0/77.1|81.6      |90.3      | 64.0     |
|ALBERT-large    |82.4      |90.6/83.9 | 82.3/79.4|83.5      |91.7      | 68.5     |
|ALBERT-xlarge   |85.5      |92.5/86.1 | 86.1/83.1|86.4      |92.4      | 74.8     |
|ALBERT-xxlarge  |91.0      |94.8/89.3 | 90.2/87.4|90.8      |96.9      | 86.5     |


### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-1909-11942,
  author    = {Zhenzhong Lan and
               Mingda Chen and
               Sebastian Goodman and
               Kevin Gimpel and
               Piyush Sharma and
               Radu Soricut},
  title     = {{ALBERT:} {A} Lite {BERT} for Self-supervised Learning of Language
               Representations},
  journal   = {CoRR},
  volume    = {abs/1909.11942},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.11942},
  archivePrefix = {arXiv},
  eprint    = {1909.11942},
  timestamp = {Fri, 27 Sep 2019 13:04:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-11942.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```