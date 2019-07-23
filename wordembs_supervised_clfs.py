# -*- coding: utf-8 -*-
import os
import sys
import pdb
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

import MeanEmbedding
from load_utils import run_classifier 
from load_utils import vocab_embed_fromfile
from load_utils import vocab_pre_word2vec
from load_utils import load_fasttext_vectors
from load_utils import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataname', default='t6', help='dataset name', 
        choices=['t6','t26','2C'])
    parser.add_argument('-c','--classifiername', default='RF', help='which classifier to use', 
        choices=['BernoulliNB','GaussianNB','RF','SVM','KNN'])
    parser.add_argument('-bow','--BagOfWordRepresentation', default='binary', 
        help='which representation-- binary bow or word embeddings bow aggregation', 
        choices=['binary','embedding'])
    parser.add_argument('-e','--embed_type', default='glove', help='word embeddings to use',
        choices=['Glove','crisisGlove','Word2Vec', 'crisisWord2Vec','FastText','crisisFastText'])
    parser.add_argument('-a','--average', default='mean', 
        help='word embeddings aggregation type',
        choices=['mean','tfidf', 'minmaxmean'])
    args = parser.parse_args()

    data_name = args.dataname # t6 or t26, 2C, 4C
    clf_name = args.classifiername  # classfier
    clf_representation = args.BagOfWordRepresentation  # binary embedding 
    embed_type = args.embed_type # glove, word2vec, fasttext
    average_type = args.average # mean, minmaxmean, tfidf (weighted)


    file_path = []
    train_list = []
    test_list = []

    if data_name == "t6":
        file_path = '../data/CrisisLexT6_cleaned/'
        disasters = ["sandy", "queensland", "boston", "west_texas", "oklahoma", "alberta"]
        test_list = ["{}_glove_token.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "t26":
        file_path = '../data/CrisisLexT26_cleaned/'
        disasters = ["2012_Colorado_wildfires", "2013_Queensland_floods", "2013_Boston_bombings", "2013_West_Texas_explosion", "2013_Alberta_floods", "2013_Colorado_floods", "2013_NY_train_crash"]
        test_list = ["{}-tweets_labeled.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "2C":
        file_path = '../data/2CTweets_cleaned/'
        disasters = ["Memphis", "Seattle", "NYC", "Chicago", "SanFrancisco", "Boston", "Brisbane", "Dublin", "London", "Sydney"]
        test_list = ["{}2C.csv.token.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}2C_training.csv".format(disaster) for disaster in disasters]

    accu_list = []
    roc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    if clf_representation == "binary":
        # Leave one out 
        for train, test in zip(train_list,test_list):
            # load train file, test file
            train_file = os.path.join(file_path,train)
            test_file = os.path.join(file_path,test)
            xtrain, ytrain = load_data(data_name, train_file)
            xtest, ytest = load_data(data_name, test_file)
            # convert tweets to a matrix of features counts
            count_vect = CountVectorizer(analyzer=str.split)
            X_train = count_vect.fit_transform(xtrain)
            X_test = count_vect.transform(xtest)
            # binarize to 0/1 representation
            binarizer = Binarizer().fit(X_train)
            X_train = binarizer.transform(X_train)
            binarizer = Binarizer().fit(X_test)
            X_test = binarizer.transform(X_test)
            # run classifier
            print(test)
            accu, roc,precision,recall,f1 = run_classifier(X_train,ytrain,X_test,ytest,clf_name,100) 
            # store results for each disaster in the lists
            accu_list.append(accu)
            roc_list.append(roc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        # print out results and average results
        print ("{}_binary_{}_LOO_accuracy {}".format(data_name, clf_name,accu_list))
        print ("{}_binary_{}_LOO_roc {}".format(data_name,clf_name,roc_list))
        print ("{}_binary_{}_LOO_precision {}".format(data_name,clf_name,precision_list))
        print ("{}_binary_{}_LOO_recall {}".format(data_name,clf_name,recall_list))
        print ("{}_binary_{}_LOO_f1 {}".format(data_name,clf_name,f1_list))
        print("{0}_binary_LOO_{1} {2:.4f} + {3:.4f} {4:.4f} + {5:.4f} {6:.4f} + {7:.4f} {8:.4f} + {9:.4f} {10:.4f} + {11:.4f}".format(data_name,clf_name,np.mean(accu_list),np.std(accu_list), np.mean(roc_list),np.std(roc_list), np.mean(f1_list),np.std(f1_list),np.mean(precision_list),np.std(precision_list),np.mean(recall_list),np.std(recall_list)))
            
    if clf_representation == "embedding":
        wordembd_files_dict = {
            'Glove':'../data/glove.twitter.27B.100d.txt',
            'crisisGlove': '../data/glove.crisis.100d.txt',
            'Word2Vec': '../data/GoogleNews-vectors-negative300.bin',
            'crisisWord2Vec': '../data/word2vec.crisis.300d.model',
            'FastText':'../data/wiki-news-300d-1M.vec',
            'crisisFastText': '../data/crisis-fasttext-model300d.vec'
            }
        wordembd_file = wordembd_files_dict[embed_type]
        word_embeddings = {}
        dimension = 100   # default for Glove type embedding

        # First read the first pair of train + test corpus to get all words 
        # that appears in dataset, for memory efficiency, 
        # only extract these words from the word embeddings files
        train_file = os.path.join(file_path,train_list[0])
        test_file = os.path.join(file_path,test_list[0])
        xtrain, ytrain = load_data(data_name, train_file)
        xtest, ytest = load_data(data_name, test_file)
        xcorpus = list(xtrain)
        xcorpus.extend(xtest)
        count_vect = CountVectorizer(analyzer=str.split)
        count_vect.fit(xcorpus)
        vocab = count_vect.vocabulary_ 

        if 'Glove' in embed_type: # read Glove or crisisGlove embeddings
            word_embeddings = vocab_embed_fromfile(wordembd_file,vocab)
        elif 'Word2Vec' in embed_type: # read pre-trained Word2Vec embeddings
            if embed_type == "Word2Vec":
                model = KeyedVectors.load_word2vec_format(wordembd_file, binary=True)
            else: # read crisisWord2Vec embeddings
                model = Word2Vec.load(wordembd_file)
            word_embeddings, dimension = vocab_pre_word2vec(model,vocab)
        elif 'FastText' in embed_type: 
            # read FastText or CrisisFastText embeddings
            word_embeddings, dimension = load_fasttext_vectors(wordembd_file,vocab)
        
        # Leave one out 
        for train, test in zip(train_list,test_list):
            train_file = os.path.join(file_path,train)
            test_file = os.path.join(file_path,test)
            xtrain, ytrain = load_data(data_name, train_file)
            xtest, ytest = load_data(data_name, test_file)
    
            analyze = count_vect.build_analyzer()
            X_train = list(map(lambda tweet:analyze(tweet), xtrain))
            X_test = list(map(lambda tweet:analyze(tweet), xtest))

            if average_type == "mean":
                X_train = MeanEmbedding.MeanEmbeddingVectorizer(word_embeddings).transform(X_train)
                X_test = MeanEmbedding.MeanEmbeddingVectorizer(word_embeddings).transform(X_test)
            elif average_type == "tfidf":
                tfidfembed = MeanEmbedding.TfidfEmbeddingVectorizer(word_embeddings).fit(X_train)
                X_train = tfidfembed.transform(X_train)
                X_test = tfidfembed.transform(X_test)
            elif average_type == "minmaxmean":
                X_train = MeanEmbedding.MinMaxMeanEmbeddingVectorizer(word_embeddings).transform(X_train)
                X_test = MeanEmbedding.MinMaxMeanEmbeddingVectorizer(word_embeddings).transform(X_test)

            print(test)
            accu, roc,precision,recall,f1 = run_classifier(X_train,ytrain,X_test,ytest,clf_name,100) 
            accu_list.append(accu)
            roc_list.append(roc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        print ("{}_{}_wordembd_{}_{}_LOO_accuracy {}".format(data_name, average_type, clf_name,embed_type+str(dimension),accu_list))
        print ("{}_{}_wordembd_{}_{}_LOO_roc {}".format(data_name,average_type, clf_name,embed_type+str(dimension),roc_list))
        print ("{}_{}_wordembd_{}_{}_LOO_precision {}".format(data_name, average_type, clf_name,embed_type+str(dimension),precision_list))
        print ("{}_{}_wordembd_{}_{}_LOO_recall {}".format(data_name,average_type, clf_name,embed_type+str(dimension),recall_list))
        print ("{}_{}_wordembd_{}_{}_LOO_f1 {}".format(data_name, average_type, clf_name,embed_type+str(dimension),f1_list))
        
        print("{0}_{1}_wordembd_LOO_{2}_{3} {4:.4f} + {5:.4f} {6:.4f} + {7:.4f} {8:.4f} + {9:.4f} {10:.4f} + {11:.4f} {12:.4f} + {13:.4f}".format(data_name,average_type,clf_name,
            embed_type+str(dimension),np.mean(accu_list),np.std(accu_list), np.mean(roc_list),np.std(roc_list), np.mean(f1_list),np.std(f1_list),np.mean(precision_list),np.std(precision_list),np.mean(recall_list),np.std(recall_list)))
    

    
if __name__ == "__main__":
    main()
