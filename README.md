# Towards Explainable Multi-Label Text Classification: A Supervised Rationalisation Framework for Identifying Indicators of Forced Labour

Rationalisation methods attempt to explain the outcome of a text classification model by providing a natural language explanation *rationale* [[1]](#1). It has been observed that rationales are more understandable and easier to use since they are verbalised in human-comprehensible natural language [[2]](#2) [[3]](#3).

This repository presents a rationalisation framework to explain the outcome of a multi-label text classifier through extractive rationales.

Our framework uses multi-task learning to produce rationales at a label level and allows the alternative of including human rationales during training as an extra supervision signal. We employ our framework to identify indicators of forced labour, as defined by the International Labour Organization [[4]](#4), for a rationale-annotated corpus of news articles [[5]](#5).

## Explainable Framework

We detail our framework for explainable text classification based on a multi-task learning implementation of the *encoder-decoder* architecture [[1]](#1) to produce rationales at a label level for a multi-label setting. The **encoder** is the module responsible for identifying the rationales within the input sequence at a label level, and the **decoder** is tasked with predicting labels based on the generated rationales [[6]](#6) [[7]](#7).

![explainable_framework](https://github.com/emendezguzman/rationalisation_framework/assets/90763977/16139184-338a-46ce-bd12-8c4471ff451c)

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

### Rationalisation Experiments

The rationalisation experiments were performed with the aid of the [EGG Toolkit](https://github.com/facebookresearch/EGG) library, a Python package that allows researchers to quickly implement multi-agent games with discrete channel communication [[9]](#9).

For the task at hand, we fine-tuned the following transformer-based models on our data set:
- **DistilBERT** [[10]](#10): A smaller and faster transformer model trained by distilling BERT base [[11]](#11). 
- **ALBERT** [[12]](#12): A light version of BERT \cite{devlin2018bert} that uses parameter-reduction techniques that allow for large-scale configurations.
- **RoBERTa** [[13]](#13): A retraining of BERT with improved architecture and training methodology.  For this model, we use the base, distil-roberta and large versions.
- **XLNet** [[14]](#14): A generalized autoregressive pre-trained method that uses improved training methodology and larger data than BERT.
- **DeBERTa** [[15]](#15): A variant of the BERT model that introduces disentangled attention mechanisms and performs dynamic weight adaptation.

For more details about the implementation please refer to **train.py**.

### Hyperparameter Tuning

We split the data set into training, validation and test sets according to a 70:10:20 ratio and search the hyperparameter values that minimise the function loss over the validation set. To optimise the training process, we tuned the model hyperparameters using a random search method and run a total of 25 training runs using WandB [[16]](#16). For more details about the implementation please refer to **hyperparameter_tuning.py**.

### Model Evaluation

The primary goal of our rationalisation framework is to simultaneously enhance predictive performance and explainability by identifying concise and relevant rationales. Consequently, we employ the following evaluation metrics:
- **Predictive Performance**: F1 Score (F1), Label Ranking Precision Average Precision Score (LRAP), and Exact Match Ratio (EMR) [[17]](#17).
- **Explainability**: Plausibility, Suffiency and Comprehensiveness [[2]](#2).

For details about the implementation please refer to **evaluation.ipynb**.

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

<a id="9">[9]</a>
Kharitonov, E., Chaabouni, R., Bouchacourt, D. and Baroni, M. (2019). 
EGG: a toolkit for research on Emergence of lanGuage in Games.
arXiv preprint arXiv:1907.00852.

<a id="10">[10]</a>
Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019).
DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
arXiv preprint arXiv:1910.01108

<a id="11">[11]</a>
Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2018).
Bert: Pre-training of deep bidirectional transformers for language understanding
arXiv preprint arXiv:1810.04805

<a id="12">[12]</a>
Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P. and Soricut, R. (2019).
Albert: A lite bert for self-supervised learning of language representations
arXiv preprint arXiv:1909.11942

<a id="13">[13]</a>
Liu, Y. et al. (2019).
Roberta: A robustly optimized bert pretraining approach
arXiv preprint arXiv:1907.11692

<a id="14">[14]</a>
Yang, Z. et al. (2019).
Xlnet: Generalized autoregressive pretraining for language understanding
Advances in neural information processing systems no. 32.

<a id="15">[15]</a>
He, P., Liu, X., Gao, J. and Chen, W. (2020). 
Deberta: Decoding-enhanced bert with disentangled attention. 
arXiv preprint arXiv:2006.03654.

<a id="16">[16]</a>
Biewald, L. (2020)
Experiment tracking with weights and biases, 
Software available from wandb.com

<a id="17">[17]</a>
Feldman, R. and Sanger, J. (2007).
The text mining handbook: advanced approaches in analyzing unstructured data
Software available from wandb.com
