# Probability Weighted Word Saliency(PWWS)

We add prediction of victim classifier from the Probability Weighted Word Saliency(PWWS)([Github](https://github.com/JHL-HUST/PWWS)). The implementations of the ACL2019 paper [Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency](https://www.aclweb.org/anthology/P19-1103).

## Overview
* `data_set/aclImdb/` , `data_set/ag_news_csv/`and`data_set/yahoo_10` are placeholder directories for the IMDB Review, AG's News and Yahoo! Answer, respectively.
* `word_level_process.py`and`char_level_process.py` contain two different prepressing methods of dataset for word-level and char-level, respectively.
* `neural_networks.py` contain implementations of four neural networks(word-based CNN, Bi-directional LSTM, char-based CNN, LSTM) used in paper.
* Use `training.py`to train four NN in `neural_networks.py`.
* `fool.py`, `evaluate_word_saliency.py`, `get_NE_list.py`,`adversarial_tools.py`and`paraphrase.py`build the experiment pipeline.

## Dependencies
* Python 3.7.1.
* Versions of all depending libraries are specified in `requirements.txt`. To reproduce the reported results, please make sure that the specified versions are installed.
* If you did not download WordNet(a lexical database for the English language), use `nltk.download('wordnet')` to do so.(Cancel the code comment on line 14 in `paraphrase. py`) 


## Usage

* Download dataset files from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) , which include
    - IMDB: `aclImdb.zip`. Decompression and place the folder`aclImdb` in`data_set/`.
    - AG's News: `ag_news_csv.zip`. Decompression and place the folder `ag_news_csv` in`data_set/`.
    - Yahoo Answers: `yahoo_10.zip`. Decompression and place the folder `yahoo_10` in`data_set/`.
* Download `glove.6B.100d.txt`from [google drive](https://drive.google.com/open?id=1YdndNH0RE6BEpg04HtK6VWemYrowWzvA) and place the file in `/`.
* Run `training.py` or use command like`python3 training.py --model word_cnn --dataset imdb --level word`. You can reset the model hyper-parameters in `neural_networks.py` and `config.py`.Note that neither this repository nor the paper provides an implementation of char_cnn on IMDB and Yahoo! Answers datasets.
* Run `fool.py` or use command like`python3 fool.py --model word_cnn --dataset imdb --level word`to generate adversarial examples using PWWS.
* If you want to train or fool different models, reset the argument in `training.py`and`fool.py`.
### Result on adversarial generation

`runs/`contains some pretrained NN models. 

`fool_result/`contains the adversarial text generation: 

- `adv.txt` means adversarial texts.
- `org.txt` means original texts.
- `adv_predict.txt` means prediction of victim classifier for adversarial texts.
- `org_predict.txt` means prediction of victim classifier for original texts.


## Contact

* If you have any questions regarding the code, please create an issue or contact the [owner](https://github.com/RenShuhuai-Andy) of this repository.

##  Acknowledgments

- Code refer to: PWWS ([Github](https://github.com/JHL-HUST/PWWS)).

