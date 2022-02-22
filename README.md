# 🌡 CsFEVER and 🌐 CTKFacts: Czech datasets for fact verification
This is a code supplement of our paper submitted to LREV in February 2022.
The full text of our paper in its latest revision, as well as its LaTeX sources, are to be found https://arxiv.org/pdf/2201.11115.pdf.
Please, cite the paper when using our datasets or models presented in this repository.

## 🗂 Contents
1. **[csfever](https://github.com/aic-factcheck/csfever)**: code used for the localization of [EnFEVER](https://fever.ai/dataset/fever.html) dataset. Follows the method described in paper section **3.1**.
2. **TODO**: DR training & eval scripts
3. **TODO**: Claimgen (DB Wrapper + dict selector) app
4. **[annotations-platform](https://github.com/aic-factcheck/csfever)**: our custom-built annotation platform, written in [Yii2](https://www.yiiframework.com/) framework, relies on a **db wrapper** application serving CTK paragraphs in parallel
6. **[dataset](https://github.com/aic-factcheck/ctkfacts)**: 🌐 CTKFacts dataset export & splitting scripts
   - **[dataset/notebooks](https://github.com/aic-factcheck/ctkfacts/notebooks)**: notebooks presenting codes used for 🌐 CTKFacts [splitting](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/datasets.ipynb), [measuring aggreement](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/agreement.ipynb) using Krippendorff's ⍺ metric and [NLI model training](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/training.ipynb) and [validation](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/validation.ipynb)
7. **TODO**: NLI training ✔️
8. **TODO**: Full pipeline testing
9. **TODO**: 🤗 dataset links: CsFEVER, CsFEVER-NLI, CtkFACTS, CtkFACTS-NLI
10. **TODO**: 🤗 transformer links: XLMR@SQUAD2&CTKFacts-NLI, XLMR@XNLI&CsFEVER

### 
