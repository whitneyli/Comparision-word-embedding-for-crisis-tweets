# -*- coding: utf-8 -*-

import sys
import pdb
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append('./SIF-master/src')
sys.path.append('../')
import data_io, params, SIF_embedding
from load_utils import load_data
from load_utils import run_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataname', default='t6', help='dataset name', 
        choices=['t6','t26','2C'])
    parser.add_argument('-c','--classifiername', default='RF', help='which classifier to use', 
        choices=['GaussianNB','RF','SVM','KNN'])
    args = parser.parse_args()
    data_name = args.dataname # t6 or t26, 2C, 4C
    clf_name = args.classifiername  # classfier
    
    # Original SIF paper used glove.840B.300d, we use the ones that were trained on twitter.
    embed_dims = [100] # can add 25, 50, 200 dimension if needed
    wordfile_list = ['../data/glove.twitter.27B.{}d.txt'.format(dim) for dim in embed_dims]
    # each line is a word and its frequency
    weightfile = 'SIF-master/auxiliary_data/enwiki_vocab_min200.txt' 
    # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    weightpara = 1e-3 
    # number of principal components to remove in SIF weighting scheme
    rmpc = 1 

    for wordfile,dim in zip(wordfile_list,embed_dims):
        # load word vectors
        (words, We) = data_io.getWordmap(wordfile)
        # load word weights
        # word2weight['str'] is the weight for the word 'str'
        word2weight = data_io.getWordWeight(weightfile, weightpara) 
        # weight4ind[i] is the weight for the i-th word
        weight4ind = data_io.getWeight(words, word2weight) 

        data_path = "../data/"
        if data_name == "t6":
            file_path = data_path + "CrisisLexT6_cleaned/"
            disasters = ["sandy", "queensland", "boston", "west_texas", "oklahoma", "alberta"]
            test_list = ["{}_glove_token.csv.unique.csv".format(disaster) for disaster in disasters]
            train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
        if data_name == "t26":
            file_path = data_path + "CrisisLexT26_cleaned/"
            disasters = ["2012_Colorado_wildfires", "2013_Queensland_floods", "2013_Boston_bombings", "2013_West_Texas_explosion", "2013_Alberta_floods", "2013_Colorado_floods", "2013_NY_train_crash"]
            test_list = ["{}-tweets_labeled.csv.unique.csv".format(disaster) for disaster in disasters]
            train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
        if data_name == "2C":
            file_path = data_path + "2CTweets_cleaned/"
            disasters = ["Memphis", "Seattle", "NYC", "Chicago", "SanFrancisco", "Boston", "Brisbane", "Dublin", "London", "Sydney"]
            test_list = ["{}2C.csv.token.csv.unique.csv".format(disaster) for disaster in disasters]
            train_list = ["{}2C_training.csv".format(disaster) for disaster in disasters]

        accu_list = []
        roc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for train, test in zip(train_list,test_list):
            train_file = os.path.join(file_path,train)
            test_file = os.path.join(file_path,test)
            xtrain, ytrain = load_data(data_name, train_file)
            xtest, ytest = load_data(data_name, test_file)

            # load train
            # xtrain_windx is the array of word indices, m_train is the binary mask indicating whether there is a word in that location
            xtrain_windx, m_train = data_io.sentences2idx(xtrain, words) 
            w_train = data_io.seq2weight(xtrain_windx, m_train, weight4ind) # get word weights

            # set parameters
            paramss = params.params()
            paramss.rmpc = rmpc
            # get SIF embedding
            train_embed = SIF_embedding.SIF_embedding(We, xtrain_windx, w_train, paramss) # embedding[i,:] is the embedding for sentence i

            # load target
            # xtest_windx is the array of word indices, m_test is the binary mask indicating whether there is a word in that location
            xtest_windx, m_test = data_io.sentences2idx(xtest, words) 
            # get word weights
            w_test = data_io.seq2weight(xtest_windx, m_test, weight4ind) 

            # set parameters
            paramsss = params.params()
            paramsss.rmpc = rmpc
            # get SIF embedding
            test_embed = SIF_embedding.SIF_embedding(We, xtest_windx, w_test, paramsss) # embedding[i,:] is the embedding for sentence i

            print (test)
            accu, roc, precision, recall, f1 = run_classifier(train_embed,ytrain,test_embed,ytest,clf_name,100)
            accu_list.append(accu)
            roc_list.append(roc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        print ("{}_SIF_{}_LOO_accuracy {}".format(data_name,clf_name+str(dim),accu_list))
        print ("{}_SIF_{}_LOO_roc {}".format(data_name,clf_name+str(dim),roc_list))
        print ("{}_SIF_{}_LOO_precision {}".format(data_name,clf_name+str(dim),precision_list))
        print ("{}_SIF_{}_LOO_recall {}".format(data_name,clf_name+str(dim),recall_list))
        print ("{}_SIF_{}_LOO_f1 {}".format(data_name,clf_name+str(dim),f1_list))
        print("{0}_SIF_LOO_{1} {2:.4f} + {3:.4f} {4:.4f} + {5:.4f} {6:.4f} + {7:.4f} {8:.4f} + {9:.4f} {10:.4f} + {11:.4f}".format(data_name,clf_name+str(dim), np.mean(accu_list),np.std(accu_list), np.mean(roc_list),np.std(roc_list),np.mean(f1_list),np.std(f1_list),np.mean(precision_list),np.std(precision_list),np.mean(recall_list),np.std(recall_list)))

if __name__ == '__main__':
    main()

