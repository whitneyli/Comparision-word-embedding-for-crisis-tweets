# -*- coding: utf-8 -*-
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from load_utils import run_classifier
from load_utils import load_data


def main():
  
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataname', default='t6', help='dataset name', 
        choices=['t6','t26','2C'])
    parser.add_argument('-c','--classifiername', default='RF', help='which classifier to use', 
        choices=['GaussianNB','RF','SVM','KNN'])
    args = parser.parse_args()
    data_name = args.dataname # t6 or t26, 2C, 4C
    clf_name = args.classifiername  # classfier
    # 
    module_name = "universal-sentence-encoder"
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"
    # hub_module_names = ['universal-sentence-encoder-large', 'Wiki-words-250-with-normalization' ]
    # module_urls = ["https://tfhub.dev/google/universal-sentence-encoder-large/2", "https://tfhub.dev/google/Wiki-words-250-with-normalization/1"]
    dataset_path = "./data/"
    disasters =[]
    train_list = []
    test_list = []
    if data_name == "t6":
        file_path = dataset_path + 'CrisisLexT6_cleaned/'
        disasters = ["sandy", "queensland", "boston", "west_texas", "oklahoma", "alberta"]
        test_list = ["{}_glove_token.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "t26":
        file_path = dataset_path + 'CrisisLexT26_cleaned/'
        disasters = ["2012_Colorado_wildfires", "2013_Queensland_floods", "2013_Boston_bombings", "2013_West_Texas_explosion", "2013_Alberta_floods", "2013_Colorado_floods", "2013_NY_train_crash"]
        test_list = ["{}-tweets_labeled.csv.unique.csv".format(disaster) for disaster in disasters]
        train_list = ["{}_training.csv".format(disaster) for disaster in disasters]
    if data_name == "2C":
        file_path = dataset_path + '2CTweets_cleaned/'
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

        xtrain, ytrain = load_data(data_name, train_file)
        xtest, ytest = load_data(data_name, test_file)
        
        embed = hub.Module(module_url)
        
        train_output = "{}{}.tfSent.train.npy".format(output_dir, disaster)
        test_output = "{}{}.tfSent.test.npy".format(output_dir, disaster)
        if not os.path.isfile(train_output):
            with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                train_embed = session.run(embed(xtrain))
                test_embed = session.run(embed(xtest))
                np.save(train_output,train_embed)
                np.save(test_output,test_embed)
        else:
            train_embed = np.load(train_output)
            test_embed = np.load(test_output)

        print(test)
        accu, roc,precision,recall,f1 = run_classifier(train_embed,ytrain,test_embed,ytest,clf_name, 100)
        accu_list.append(accu)
        roc_list.append(roc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print ("{}_tfSent_{}_{}_LOO_accuracy {}".format(data_name,module_name, clf_name,accu_list))
    print ("{}_tfSent_{}_{}_LOO_roc {}".format(data_name,module_name,clf_name,roc_list))
    print ("{}_tfSent_{}_{}_LOO_precision {}".format(data_name,module_name,clf_name,precision_list))
    print ("{}_tfSent_{}_{}_LOO_recall {}".format(data_name,module_name,clf_name,recall_list))
    print ("{}_tfSent_{}_{}_LOO_f1 {}".format(data_name,module_name,clf_name,f1_list))
    print("{0}_tfSent_LOO_{1} {2:.4f} + {3:.4f} {4:.4f} + {5:.4f} {6:.4f} + {7:.4f} {8:.4f} + {9:.4f} {10:.4f} + {11:.4f}".format(data_name+module_name,clf_name,
        np.mean(accu_list),np.std(accu_list), np.mean(roc_list),np.std(roc_list),np.mean(f1_list),np.std(f1_list),np.mean(precision_list),np.std(precision_list),np.mean(recall_list),np.std(recall_list)))

if __name__ == '__main__':
    main()
