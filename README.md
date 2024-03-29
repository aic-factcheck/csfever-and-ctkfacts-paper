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
7. 🤗 transformer links: [XLMR@SQUAD2&CTKFacts-NLI](https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-ctkfacts), [XLMR@XNLI&CsFEVER](https://huggingface.co/ctu-aic/xlm-roberta-large-xnli-csfever)

### 🤗 Datasets
Published datasets ([🌡 CsFEVER](https://huggingface.co/datasets/ctu-aic/csfever), [🌡 CsFEVER-NLI](https://huggingface.co/datasets/ctu-aic/csfever_nli), [🌐 CtkFACTS-NLI](https://huggingface.co/datasets/ctu-aic/ctkfacts_nli) and [🌐 CtkFACTS](https://huggingface.co/datasets/ctu-aic/ctkfacts)) are released on our [🤗 Huggingface dataset repository](https://huggingface.co/ctu-aic) and referred to in this codebase using git submodules nested in the **[datasets](datasets)** directory

### 🤖 Models
The DR and NLI models we established as baselines for our datasets are to be found in on our [🤗 Huggingface model repository](https://huggingface.co/ctu-aic) and can be navigated using the following map

#### 🔎 DR models
- Todo
#### ⊨ NLI models (two best-performing for each dataset, benchmark in paper)
- https://huggingface.co/ctu-aic/xlm-roberta-large-xnli-ctkfacts_nli 🌐 **XLM-RoBERTa@XNLI@CTKFacts-NLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-ctkfacts_nli 🌐 **XLM-RoBERTa@XNLI@CTKFacts-NLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-csfever_nearestp 🌡 **XLM-RoBERTa@SQUAD2@CsFEVER (*NearestP*)**
- https://huggingface.co/ctu-aic/bert-base-multilingual-cased-csfever_nearestp  🌡 **M-BERT@CsFEVER (*NearestP*)**
- https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-csfever_nli 🤒 **XLM-RoBERTa@SQUAD2@CsFEVER-NLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-xnli-csfever_nli  🤒 **XLM-RoBERTa@XNLI@CsFEVER-NLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-xnli-enfever_nli 🇬🇧 **XLM-RoBERTa@XNLI@EnFEVER-NLI**
- https://huggingface.co/ctu-aic/xlm-roberta-large-squad2-enfever_nli 🇬🇧 **XLM-RoBERTa@SQUAD2@EnFEVER-NLI**
