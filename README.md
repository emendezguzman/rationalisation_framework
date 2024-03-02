# Towards Explainable Multi-Label Text Classification: A Supervised Rationalisation Framework for Identifying Indicators of Forced Labour

Rationalisation methods attempt to explain the outcome of a text classification model by providing a natural language explanation *rationale* [[1]](#1). It has been observed that rationales are more understandable and easier to use since they are verbalised in human-comprehensible natural language [[2]](#2) [[3]](#3).

This repository presents a rationalisation framework to explain the outcome of a multi-label text classifier through extractive rationales.

Our framework uses multi-task learning to produce rationales at a label level and allows the alternative of including human rationales during training as an extra supervision signal. We employ our framework to identify indicators of forced labour, as defined by the International Labour Organization [[4]](#4), for a rationale-annotated corpus of news articles [[5]](#5).

## Explainable Framework

We detail our framework for explainable text classification based on a multi-task learning implementation of the *encoder-decoder* architecture [[1]](#1) to produce rationales at a label level for a multi-label setting. The **encoder** is the module responsible for identifying the rationales within the input sequence at a label level, and the **decoder** is tasked with predicting labels based on the generated rationales [[6]](#6) [[7]](#7).

## Corpus

The corpus consists of **989 news articles** retrieved from specialised data sources and annotated according to the annotation schema detailed in the Annotation Guidelines.

The JSON file is structured as follows:

- **Index**: News article ID (Integer).
- **Title**: News article title (String).
- **Content**: News article content (String).
- **Labels**: List of labels suitable for multi-class multi-label classification.
- **Set**: String detailing the role of the example in classification experiments (train, validation and test set).
- **Rationales**: List of rationales, specifyng both the text and label for which the rationales were provided.

## Rationalisation

### Installation

You need to have Python 3.7 [[8]](#8) or higher installed. It is recommended that you use a virtual environment:

```
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./my_venv
source ./my_venv/bin/activate
```

Install all required Python packages using:

```
pip install -r requirements.txt
```

Clone the repository:

```
git clone https://github.com/emendezguzman/rationalisation_framework.git
```

### Classification Experiments

The classification experiments were performed with the aid of the [Simple Transformers](https://simpletransformers.ai/) library, a Python package based on the Transformers library by HuggingFace [[5]](#5).

For the task at hand, we fine-tuned the following transformer-based models on our data set:
- **DistilBERT** [[6]](#6): A smaller and faster transformer model trained by distilling BERT base [[7]](#7). 
- **ALBERT** [[8]](#8): A light version of BERT \cite{devlin2018bert} that uses parameter-reduction techniques that allow for large-scale configurations.
- **RoBERTa** [[9]](#9): A retraining of BERT with improved architecture and training methodology.  For this model, we use the base, distil-roberta and large versions.
- **XLNet** [[10]](#10): A generalized autoregressive pre-trained method that uses improved training methodology and larger data than BERT [[7]](#7).

### Hyperparameter Tuning

We split the data set into training, validation and test sets according to a 70:10:20 ratio and search the hyperparameter values that minimise the function loss over the validation set. To optimise the training process, we tuned the model hyperparameters using a random search method and run a total of 40 training runs using WandB [[11]](#11). For more details about the implementation please refer to **hyperparameter_tuning.ipynb**.

### Model Evaluation

Finally, we employed four metrics to evaluate the performance of our baseline classifiers: F1 Score (F1), Label Ranking Precision Average Precision Score (LRAP), and Exact Match Ratio (EMR) [[12]](#12). For details about the implementation please refer to **evaluation.ipynb**.

## References

<a id="1">[1]</a> 
Lei, T., Barzilay, R. and Jaakkola, T. (2016)
Rationalizing neural predictions.
arXiv preprint arXiv:1606.04155.

<a id="2">[2]</a> 
DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2019).
ERASER: A benchmark to evaluate rationalized NLP models.
arXiv preprint arXiv:1911.03429.

<a id="3">[3]</a> 
Wang, H. and Dou, Y. (2022)
Recent Development on Extractive Rationale for Model Interpretability: A Survey. 
In: 2022 International Conference on Cloud Computing, Big Data and Internet of Things (3CBIT) (pp. 354-358). IEEE.

<a id="4">[4]</a>
International Labour Organization. (2012). 
ILO Indicators of Forced Labour. 
In: Special Action Programme to Combat Forced Labour. Special Action Programme to Combat Forced Labour.

<a id="5">[5]</a>
Guzman, E. M., Schlegel, V., & Batista-Navarro, R. (2022). 
RaFoLa: A Rationale-Annotated Corpus for Detecting Indicators of Forced Labour. 
arXiv preprint arXiv:2205.02684.

<a id="6">[6]</a>
Bastings, J., Aziz, W. and Titov, I. (2019). 
Interpretable neural predictions with differentiable binary variables. 
arXiv preprint arXiv:1905.08160.

<a id="7">[7]</a>
Madani, M. R. G., & Minervini, P. (2023). 
REFER: An End-to-end Rationale Extraction Framework for Explanation Regularization. 
arXiv preprint arXiv:2310.14418.

<a id="8">[8]</a>
Van Rossum, G. and Drake, F. (2009).
Python 3 Reference Manual
CreateSpace.

<a id="5">[5]</a>
Wolf, T. et al. (2019).
Huggingface's transformers: State-of-the-art natural language processing
arXiv preprint arXiv:1910.03771

<a id="6">[6]</a>
Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019).
DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
arXiv preprint arXiv:1910.01108

<a id="7">[7]</a>
Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2018).
Bert: Pre-training of deep bidirectional transformers for language understanding
arXiv preprint arXiv:1810.04805

<a id="8">[8]</a>
Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P. and Soricut, R. (2019).
Albert: A lite bert for self-supervised learning of language representations
arXiv preprint arXiv:1909.11942

<a id="9">[9]</a>
Liu, Y. et al. (2019).
Roberta: A robustly optimized bert pretraining approach
arXiv preprint arXiv:1907.11692

<a id="10">[10]</a>
Yang, Z. et al. (2019).
Xlnet: Generalized autoregressive pretraining for language understanding
Advances in neural information processing systems no. 32.

<a id="11">[11]</a>
Biewald, L. (2020)
Experiment tracking with weights and biases, 
Software available from wandb.com

<a id="12">[12]</a>
Feldman, R. and Sanger, J. (2007).
The text mining handbook: advanced approaches in analyzing unstructured data
Software available from wandb.com
