# Comparison of Word Embeddings and Sentence Encodings as Generalized Representations for Crisis Tweet Classification Tasks

This repo contains the python 3.7 scripts for paper [Comparison of Word Embeddings and Sentence Encodings as Generalized Representations for Crisis Tweet Classification Tasks](https://pdfs.semanticscholar.org/6787/1d198386f513f86c71cc603d993b85d5709f.pdf).

## Requriments

Require numpy, pandas, scikit-learn, gensim to run word embedding related experiments. For other sentence encodings, [SIF](https://github.com/PrincetonML/SIF) additionally requires theano and lasagne, [InferSent](https://github.com/facebookresearch/InferSent) additionally requires pytorch and nltk, and Universal encoding ([tfSent](https://tfhub.dev/google/universal-sentence-encoder/1)) from Tensorflow requires Tensorflow.

## Data
Preprocessed data files are in ''data'' directory.


## Get started
*To run word embeddings related experiments:* 

0) Download word embeddings files at [link](http://people.beocat.ksu.edu/~hongminli/) to ''data'' directory .

1) Run wordembs\_supervised\_clfs.py:
```bash
python wordembs_supervised_clfs.py -h
```

usage: wordembs\_supervised\_clfs.py [-h] 

                                   [-d {t6,t26,2C}]
                                   [-c {BernoulliNB,GaussianNB,RF,SVM,KNN}]
                                   [-bow {binary,embedding}]
                                   [-e {Glove,crisisGlove,Word2Vec,crisisWord2Vec,FastText,crisisFastText}]
                                   [-a {mean,tfidf,minmaxmean}]

Example:
```
python wordembs_supervised_clfs.py -d t26 -c GaussianNB -bow embedding -e Glove -a mean
```

NOTE that BernoulliNB classifier is only makes sense when using binary bag-of-word representations.

## Sentence Encodings related
To run sentence encodings related experiments:

 **SIF**:
>
    cd SIF_sentence
    python SIF_sentence.py -h

Example:
```
python SIF_sentence.py -d t26 -c GaussianNB
```

 **InferSent**:

```bash
cd InferSent
```

1) Download [GloVe](https://nlp.stanford.edu/projects/glove/) (V1):
```bash
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

2) Download our InferSent models (V1 trained with GloVe):
```bash
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```
Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer).

3) Make sure you have the NLTK tokenizer by running the following once:
```python
import nltk
nltk.download('punkt')
```

4) Run inferSent\_crisis\_LOO.py:
```
python inferSent_crisis_LOO.py -h
usage: inferSent_crisis_LOO.py [-h] [-d {t6,t26,2C}]
                               [-c {GaussianNB,RF,SVM,KNN}]
```

Example:

```
python inferSent_crisis_LOO.py -d t26 -c RF
```

*NOTE:* The original paper was using an old version InferSent pretrained model with GloVe, so the results are slightly different. Interesting users can also try their V2 version model trained with InferSent. For details go to [InferSent](https://github.com/facebookresearch/InferSent).


**tfSent**


run tf\_sentence.py
```
python tf_sentence.py -h
usage: tf_sentence.py [-h] [-d {t6,t26,2C}] [-c {GaussianNB,RF,SVM,KNN}]
```
Example:
```
 python tf_sentence.py -d t26 -c RF
```


## References
For more details and full experimental results, see [the paper](https://pdfs.semanticscholar.org/6787/1d198386f513f86c71cc603d993b85d5709f.pdf).

```
@article{li2018comparison,
  title={Comparison of Word Embeddings and Sentence Encodings as Generalized Representations for Crisis Tweet Classification Tasks},
  author={Li, Hongmin and Li, Xukun and Caragea, Doina and Caragea, Cornelia},
booktitle={Proceedings of the ISCRAM Asian Pacific 2018 Conference – Wellington, New Zealand, November 2018},
year={2018}
}
```
