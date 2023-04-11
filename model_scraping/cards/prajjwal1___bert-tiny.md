The following model is a Pytorch pre-trained model obtained from converting Tensorflow checkpoint found in the [official Google BERT repository](https://github.com/google-research/bert). 

This is one of the smaller pre-trained BERT variants, together with [bert-mini](https://huggingface.co/prajjwal1/bert-mini) [bert-small](https://huggingface.co/prajjwal1/bert-small) and [bert-medium](https://huggingface.co/prajjwal1/bert-medium). They were introduced in the study `Well-Read Students Learn Better: On the Importance of Pre-training Compact Models` ([arxiv](https://arxiv.org/abs/1908.08962)), and ported to HF for the study `Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics` ([arXiv](https://arxiv.org/abs/2110.01518)). These models are supposed to be trained on a downstream task.

If you use the model, please consider citing both the papers:
```
@misc{bhargava2021generalization,
      title={Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics}, 
      author={Prajjwal Bhargava and Aleksandr Drozd and Anna Rogers},
      year={2021},
      eprint={2110.01518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{DBLP:journals/corr/abs-1908-08962,
  author    = {Iulia Turc and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {Well-Read Students Learn Better: The Impact of Student Initialization
               on Knowledge Distillation},
  journal   = {CoRR},
  volume    = {abs/1908.08962},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.08962},
  eprinttype = {arXiv},
  eprint    = {1908.08962},
  timestamp = {Thu, 29 Aug 2019 16:32:34 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1908-08962.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
Config of this model:
- `prajjwal1/bert-tiny` (L=2, H=128) [Model Link](https://huggingface.co/prajjwal1/bert-tiny)


Other models to check out:
- `prajjwal1/bert-mini` (L=4, H=256) [Model Link](https://huggingface.co/prajjwal1/bert-mini)
- `prajjwal1/bert-small` (L=4, H=512) [Model Link](https://huggingface.co/prajjwal1/bert-small)
- `prajjwal1/bert-medium` (L=8, H=512) [Model Link](https://huggingface.co/prajjwal1/bert-medium)

Original Implementation and more info can be found in [this Github repository](https://github.com/prajjwal1/generalize_lm_nli).

Twitter: [@prajjwal_1](https://twitter.com/prajjwal_1)