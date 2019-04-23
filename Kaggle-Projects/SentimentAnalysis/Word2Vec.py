# -*- UTF-8 -*-
#  LogisticRegression 85.52%

import re

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def clean_sentences(sentence):
    # 去除HTML Tag
    review = BeautifulSoup(sentence, features="html5lib").get_text()
    # 去除标点
    review = re.sub("[^a-zA-Z]", " ", review)
    # 所有字母小写
    review = review.lower()
    # 将句子分割成单词列表
    words_lists = review.split()

    # return like ['word0', 'word1', ...]
    return words_lists


def review2sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review)
    sentences = []
    for sentence in raw_sentences:
        if len(sentence) > 0:
            sentences.append(clean_sentences(sentence))

    # [['word0', 'word1'..], ['word0', 'word1'..], ..]
    return sentences

def word2vec_model(dataset_df_list):
    all_sentences = []
    # 将文本分割成句子
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for df in dataset_df_list:
        for review in df['review']:
            all_sentences += review2sentences(review, tokenizer)

    num_features = 300  # 词向量维度
    min_word_count = 40  # 词数目
    num_workers = 4  # 并行线程
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)

    model = Word2Vec(all_sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    # model.wv.doesnt_match("man woman child kitchen".split())
    # model.wv.doesnt_match("france england germany berlin".split())
    # model.wv.doesnt_match("paris berlin london austria".split())
    # model.wv.most_similar("man")
    # model.wv.most_similar("queen")
    # model.wv.most_similar("awful")
    return model

# 根据训练好的model，将评论也转化成定长的向量
# 一种简单的方法就是：将review中的所有单词的vector整合起来，求个平均值，代表review vector
def word_list2vec(word_list, model, feature_num):
    vec = np.zeros((feature_num,), dtype='float32')
    vocabulary_set = set(model.wv.index2word)
    num = 0
    for word in word_list:
        if word in vocabulary_set:
            vec = np.add(vec, model[word])
            num += 1
    vec = np.divide(vec, num)
    return vec


def review2word_list(raw_review):
    # 去除HTML Tag
    review = BeautifulSoup(raw_review).get_text()
    # 去除标点
    review = re.sub("[^a-zA-Z]", " ", review)
    # 所有字母小写
    review = review.lower()
    # 将句子分割成单词列表
    words_lists = review.split()
    stopwords_set = set(stopwords.words('english'))
    review = []
    # 去除停用词
    for word in words_lists:
        if word not in stopwords_set:
            review.append(word)

    # Like: ['word1', 'word2', 'word3'...]
    return review


def review_list2vec_array(review_list, model, feature_num):
    vec_array = np.zeros((len(review_list), feature_num), dtype="float32")

    for i, review in enumerate(review_list):
        vec_array[i] = word_list2vec(review2word_list(review), model, feature_num)

    return vec_array


def predict(df_train, df_test, vec_train, vec_test):
    # clf = RandomForestClassifier()
    clf = LogisticRegression()
    clf = clf.fit(vec_train, df_train['sentiment'])

    test_label = clf.predict(vec_test)
    print("Score:" + str(clf.score(vec_train, df_train['sentiment'])))
    submission = pd.DataFrame(data={'id': df_test['id'], 'sentiment': test_label})
    submission.to_csv("submission.csv")
    return submission

if __name__ == '__main__':
    train_df = pd.read_csv("data/labeledTrainData.tsv", sep="\t", encoding="latin-1")
    train_df2 = pd.read_csv("data/unlabeledTrainData.tsv", sep="\t", encoding="latin-1")
    test_df = pd.read_csv("data/testData.tsv", sep="\t", encoding="latin-1")
    model = word2vec_model([train_df, train_df2, test_df])

    vec_array = review_list2vec_array(train_df['review'], model, 300)

    test_vec_array = review_list2vec_array(test_df['review'], model, 300)

