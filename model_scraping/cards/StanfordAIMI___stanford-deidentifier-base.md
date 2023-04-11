Stanford de-identifier was trained on a variety of radiology and biomedical documents with the goal of automatising the de-identification process while reaching satisfactory accuracy for use in production. Manuscript in-proceedings. 

These model weights are the recommended ones among all available deidentifier weights.

Associated github repo: https://github.com/MIDRC/Stanford_Penn_Deidentifier

## Citation

```bibtex
@article{10.1093/jamia/ocac219,
    author = {Chambon, Pierre J and Wu, Christopher and Steinkamp, Jackson M and Adleberg, Jason and Cook, Tessa S and Langlotz, Curtis P},
    title = "{Automated deidentification of radiology reports combining transformer and “hide in plain sight” rule-based methods}",
    journal = {Journal of the American Medical Informatics Association},
    year = {2022},
    month = {11},
    abstract = "{To develop an automated deidentification pipeline for radiology reports that detect protected health information (PHI) entities and replaces them with realistic surrogates “hiding in plain sight.”In this retrospective study, 999 chest X-ray and CT reports collected between November 2019 and November 2020 were annotated for PHI at the token level and combined with 3001 X-rays and 2193 medical notes previously labeled, forming a large multi-institutional and cross-domain dataset of 6193 documents. Two radiology test sets, from a known and a new institution, as well as i2b2 2006 and 2014 test sets, served as an evaluation set to estimate model performance and to compare it with previously released deidentification tools. Several PHI detection models were developed based on different training datasets, fine-tuning approaches and data augmentation techniques, and a synthetic PHI generation algorithm. These models were compared using metrics such as precision, recall and F1 score, as well as paired samples Wilcoxon tests.Our best PHI detection model achieves 97.9 F1 score on radiology reports from a known institution, 99.6 from a new institution, 99.5 on i2b2 2006, and 98.9 on i2b2 2014. On reports from a known institution, it achieves 99.1 recall of detecting the core of each PHI span.Our model outperforms all deidentifiers it was compared to on all test sets as well as human labelers on i2b2 2014 data. It enables accurate and automatic deidentification of radiology reports.A transformer-based deidentification pipeline can achieve state-of-the-art performance for deidentifying radiology reports and other medical documents.}",
    issn = {1527-974X},
    doi = {10.1093/jamia/ocac219},
    url = {https://doi.org/10.1093/jamia/ocac219},
    note = {ocac219},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocac219/47220191/ocac219.pdf},
}
```