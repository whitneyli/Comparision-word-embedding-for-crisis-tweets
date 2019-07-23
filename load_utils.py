import io
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,precision_score,recall_score

def load_data(data_name, data_file):
    x = []
    y = []
    if data_name == "t6":
        x, y = load_file_binary_class(data_file,"tweet_cleaned"," label", "on-topic", "off-topic")
    if data_name == "t26":
        x, y = load_file_binary_class(data_file,"tweet_cleaned"," Informativeness", "Related and informative","Related - but not informative")
    if data_name == "2C":
        x,y  = load_file_binary_class(data_file,"tweet_cleaned","label", "YES","NO") 
    return x, y

def load_file_binary_class(csv_file,tweet_column,label_column,positive_class,negative_class):
    df0 = pd.read_csv(csv_file)
    df = df0.drop_duplicates(subset=tweet_column,keep='first')
    label_list = [positive_class,negative_class]
    df_labeled = df.loc[df[label_column].isin(label_list)]
    X = df_labeled[tweet_column].str.strip().tolist()
    y = []
    label_numberize = range(len(label_list))
    label_dict = dict()
    for label in label_list:
        if label == positive_class:
            label_dict[label] = 1
        else:
            label_dict[label] = 0

    y = df_labeled[label_column].str.strip().apply(lambda y: label_dict[y]).tolist()
    return X,y

def vocab_embed_fromfile(glove_vectors_file, vocab):
    word_embeddings = {}
    with open(glove_vectors_file, 'r') as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                # word_embeddings[vals[0]] = map(float, vals[1:])
                word_embeddings[vals[0]] = np.array(vals[1:], dtype=np.float32)
    return word_embeddings              

def vocab_pre_word2vec(model, vocab):
    word_embeddings = {}
    google_vocab = list(model.wv.vocab)
    dimension = len(model.wv[google_vocab[0]])
    for word in google_vocab:
        if word in vocab:
            word_embeddings[word] = model.wv[word]
    return word_embeddings, dimension

## https://fasttext.cc/docs/en/english-vectors.html
def load_fasttext_vectors(fname,vocab):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, dimension = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab:
            # data[tokens[0]] = map(float, tokens[1:])
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    return data, dimension


def run_classifier(X_train,y_train,X_test,y_test,clf_name,num_trees):
    accu = 0.0
    roc = 0.0
    predicted = []
    clf = ''
    if clf_name == "BernoulliNB":
        clf = BernoulliNB().fit(X_train,y_train)
    if clf_name == "GaussianNB":
        clf = GaussianNB().fit(X_train,y_train)
    if clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=num_trees)
        clf = clf.fit(X_train,y_train)
    if clf_name == "SVM":
        clf = SVC(cache_size=2000,probability=False)
        clf.fit(X_train,y_train)
    if clf_name == "KNN":
        n_neighbors = 5 # default is 5
        clf = neighbors.KNeighborsClassifier(n_jobs=-1)
        clf.fit(X_train,y_train)
        
    predicted = clf.predict(X_test)
    if clf_name == "SVM":
        predicted_prob = clf.decision_function(X_test)
        accu = accuracy_score(y_test,predicted)
        roc = roc_auc_score(y_test,predicted_prob)
    else:
        predicted_prob = clf.predict_proba(X_test)
        accu = accuracy_score(y_test,predicted)
        roc = roc_auc_score(y_test,predicted_prob[:,1])
    pos_presicion = precision_score(y_test,predicted)
    pos_recall =  recall_score(y_test,predicted)
    pos_f1 = f1_score(y_test,predicted)
    print("Correctly Classified: {}".format(accu))
    print(classification_report(y_test,predicted,digits=4))

    return accu, roc, pos_presicion, pos_recall, pos_f1