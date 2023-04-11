### eng-spa

* source group: English 
* target group: Spanish 
*  OPUS readme: [eng-spa](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/eng-spa/README.md)

*  model: transformer
* source language(s): eng
* target language(s): spa
* model: transformer
* pre-processing: normalization + SentencePiece (spm32k,spm32k)
* download original weights: [opus-2020-08-18.zip](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus-2020-08-18.zip)
* test set translations: [opus-2020-08-18.test.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus-2020-08-18.test.txt)
* test set scores: [opus-2020-08-18.eval.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus-2020-08-18.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| newssyscomb2009-engspa.eng.spa 	| 31.0 	| 0.583 |
| news-test2008-engspa.eng.spa 	| 29.7 	| 0.564 |
| newstest2009-engspa.eng.spa 	| 30.2 	| 0.578 |
| newstest2010-engspa.eng.spa 	| 36.9 	| 0.620 |
| newstest2011-engspa.eng.spa 	| 38.2 	| 0.619 |
| newstest2012-engspa.eng.spa 	| 39.0 	| 0.625 |
| newstest2013-engspa.eng.spa 	| 35.0 	| 0.598 |
| Tatoeba-test.eng.spa 	| 54.9 	| 0.721 |


### System Info: 
- hf_name: eng-spa

- source_languages: eng

- target_languages: spa

- opus_readme_url: https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/eng-spa/README.md

- original_repo: Tatoeba-Challenge

- tags: ['translation']

- languages: ['en', 'es']

- src_constituents: {'eng'}

- tgt_constituents: {'spa'}

- src_multilingual: False

- tgt_multilingual: False

- prepro:  normalization + SentencePiece (spm32k,spm32k)

- url_model: https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus-2020-08-18.zip

- url_test_set: https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus-2020-08-18.test.txt

- src_alpha3: eng

- tgt_alpha3: spa

- short_pair: en-es

- chrF2_score: 0.721

- bleu: 54.9

- brevity_penalty: 0.978

- ref_len: 77311.0

- src_name: English

- tgt_name: Spanish

- train_date: 2020-08-18 00:00:00

- src_alpha2: en

- tgt_alpha2: es

- prefer_old: False

- long_pair: eng-spa

- helsinki_git_sha: d2f0910c89026c34a44e331e785dec1e0faa7b82

- transformers_git_sha: f7af09b4524b784d67ae8526f0e2fcc6f5ed0de9

- port_machine: brutasse

- port_time: 2020-08-24-18:20