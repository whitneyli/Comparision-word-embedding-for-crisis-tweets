# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import nltk
sys.path.append('../')
from load_utils import run_classifier
from load_utils import load_data
sys.path.append('./InferSent-master')
from models import InferSent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataname', default='t6', help='dataset name', 
        choices=['t6','t26','2C'])
    parser.add_argument('-c','--classifiername', default='RF', help='which classifier to use', 
        choices=['GaussianNB','RF','SVM','KNN'])
    args = parser.parse_args()
    data_name = args.dataname # t6 or t26, 2C, 4C
    clf_name = args.classifiername  # classfier

    GLOVE_PATH = 'GloVe/glove.840B.300d.txt'
    dataset = '../data/'
    disasters = []
    train_list = []
    test_list = []
    if data_name == "t6":
        file_path = dataset + 'CrisisLexT6_cleaned/'
        disasters = ["sandy", "queensland", "boston", "west_texas", "oklahoma", "alberta"]
        test_list = ["{}_glove_token.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "t26":
        file_path = dataset + 'CrisisLexT26_cleaned/'
        disasters = ["2012_Colorado_wildfires", "2013_Queensland_floods", "2013_Boston_bombings", "2013_West_Texas_explosion", "2013_Alberta_floods", "2013_Colorado_floods", "2013_NY_train_crash"]
        test_list = ["{}-tweets_labeled.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "2C":
        file_path = dataset + '2CTweets_cleaned/' 
        disasters = ["Memphis", "Seattle", "NYC", "Chicago", "SanFrancisco", "Boston", "Brisbane", "Dublin", "London", "Sydney"]
        test_list = ["{}2C.csv.token.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}2C_training.csv".format(disaster) for disaster in disasters]
    
    accu_list = []
    roc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    output_dir = ''
    for disaster, train, test in zip(disasters,train_list,test_list):
        train_file = os.path.join(file_path,train)
        test_file = os.path.join(file_path,test)
        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        xtrain, ytrain = load_data(data_name, train_file)
        xtest, ytest = load_data(data_name, test_file)

        train_output = "{}{}.train.npy".format(output_dir,disaster)
        test_output = "{}{}.test.npy".format(output_dir,disaster)
        if not os.path.isfile(train_output):
            # Load our pre-trained model (in encoder/):
            V = 1
            MODEL_PATH = 'encoder/infersent%s.pkl' % V
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
            infersent = InferSent(params_model)
            infersent.load_state_dict(torch.load(MODEL_PATH))
            # Set word vector path for the model:
            W2V_PATH = './GloVe/glove.840B.300d.txt'
            infersent.set_w2v_path(W2V_PATH)
            # # Build the vocabulary of word vectors (i.e keep only those needed):
            # infersent.build_vocab(sentences, tokenize=True)
            infersent.build_vocab_k_words(K=100000)
            # Encode your sentences (list of n sentences):
            train_embed = infersent.encode(xtrain, bsize=128,tokenize=True,verbose=True)
            np.save(train_output, train_embed)
            test_embed = infersent.encode(xtest, bsize=128,tokenize=True,verbose=True)
            np.save(test_output, test_embed)
            print('file saved')
        else:
            train_embed = np.load(train_output)
            test_embed = np.load(test_output)
        
        print (test)
        accu, roc,precision,recall,f1 = run_classifier(train_embed,ytrain,test_embed,ytest,clf_name,100) 
        # print accu, roc
        accu_list.append(accu)
        roc_list.append(roc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print ("{}_InferSent_{}_LOO_accuracy {}".format(data_name,clf_name,accu_list))
    print ("{}_InferSent_{}_LOO_roc {}".format(data_name,clf_name,roc_list))
    print ("{}_InferSent_{}_LOO_percision {}".format(data_name,clf_name,precision_list))
    print ("{}_InferSent_{}_LOO_recall {}".format(data_name,clf_name,recall_list))
    print ("{}_InferSent_{}_LOO_f1 {}".format(data_name,clf_name,f1_list))
    print("{0}_InferSent_LOO_{1} {2:.4f} + {3:.4f} {4:.4f} + {5:.4f} {6:.4f} + {7:.4f} {8:.4f} + {9:.4f} {10:.4f} + {11:.4f} ".format(data_name,clf_name,
        np.mean(accu_list),np.std(accu_list), np.mean(roc_list),np.std(roc_list), np.mean(f1_list),np.std(f1_list),np.mean(precision_list),np.std(precision_list),np.mean(recall_list),np.std(recall_list)))

if __name__ == '__main__':
    main()
