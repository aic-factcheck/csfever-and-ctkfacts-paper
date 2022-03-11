# 🌡 CsFEVER and 🌐 CTKFacts: Czech datasets for fact verification
This is a code supplement of our paper submitted to LREV in February 2022.
The full text of our paper in its latest revision, as well as its LaTeX sources, are to be found https://arxiv.org/pdf/2201.11115.pdf.
Please, cite the paper when using our datasets or models presented in this repository.

## 🗂 Contents
1. **[csfever](https://github.com/aic-factcheck/csfever)**: code used for the localization of 🌡 [EnFEVER](https://fever.ai/dataset/fever.html) dataset. Follows the method described in paper section **3.1**.
2. **TODO**: DR training & eval scripts
3. **[dictionary-generation](https://github.com/aic-factcheck/dictionary-generation)**: Dictionary generation (and paragraph sampling) Flask app
4. **[annotations-platform](https://github.com/aic-factcheck/csfever)**: our custom-built annotation platform, written in [Yii2](https://www.yiiframework.com/) framework, relies on a **db wrapper** application serving CTK paragraphs in parallel
5. **[dataset](https://github.com/aic-factcheck/ctkfacts)**: 🌐 CTKFacts dataset export & splitting scripts
   - **[dataset/notebooks](https://github.com/aic-factcheck/ctkfacts/notebooks)**: notebooks presenting codes used for 🌐 CTKFacts [splitting](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/datasets.ipynb), [measuring aggreement](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/agreement.ipynb) using Krippendorff's ⍺ metric and [NLI model training](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/training.ipynb) and [validation](https://github.com/aic-factcheck/ctkfacts/blob/c62ae4373bc2332cbc29dfe8b6b356348558b476/notebooks/validation.ipynb)
6. **TODO**: Full pipeline testing
7.  **TODO**: 🤗 transformer links: XLMR@SQUAD2&CTKFacts-NLI, XLMR@XNLI&CsFEVER

### 🤗 Datasets
Published datasets ([🌡 CsFEVER](https://huggingface.co/datasets/ctu-aic/csfever), [🌡 CsFEVER-NLI](https://huggingface.co/datasets/ctu-aic/csfever_nli), [🌐 CtkFACTS-NLI](https://huggingface.co/datasets/ctu-aic/ctkfacts_nli)) are released on our [🤗 Huggingface dataset repository](https://huggingface.co/ctu-aic) and referred to in this codebase using git submodules nested in the **[datasets](datasets)** directory

### 🤖 Models
The DR and NLI models we established as baselines for our datasets are to be found in on our [🤗 Huggingface model repository](https://huggingface.co/ctu-aic) and can be navigated using the following map

#### 🔎 DR models
- Todo
#### ⊨ NLI models
- https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-ctkfacts 🌐 **XLM-RoBERTa@SQuAD2@CtkfactsNLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-xnli-csfever 🌡 **XLM-RoBERTa@XNLI@CsFEVER**
