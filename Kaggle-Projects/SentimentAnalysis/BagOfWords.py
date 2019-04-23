# -*- UTF-8 -*-
# RandomForestClassifier n-gram(1, 2)
# LogisticRegression n-gram(1, 1) 85%

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def clean_review(raw_review):
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

    for word in words_lists:
        if word not in stopwords_set:
            review.append(word)

    return " ".join(review)


def extract_features(dataset_df):
    cleaned_reviews = []
    for i in range(dataset_df.index.size):
        cleaned_reviews.append(clean_review(dataset_df.loc[i, 'review']))

    vectorizer = CountVectorizer(lowercase=None, ngram_range=(1, 2), stop_words=None, max_features=5000)
    # vectorizer = TfidfVectorizer(lowercase=None, ngram_range=(1, 1), stop_words=None, max_features=5000)
    features = vectorizer.fit_transform(cleaned_reviews)

    features = features.toarray()

    return features


def predict(df_train, df_test):
    train_review_features = extract_features(df_train)
    test_review_features = extract_features(df_test)
    clf = RandomForestClassifier()
    # clf = LogisticRegression()
    clf = clf.fit(train_review_features, df_train['sentiment'])

    test_label = clf.predict(test_review_features)
    print("Score:" + str(clf.score(train_review_features, df_train['sentiment'])))
    submission = pd.DataFrame(data={'id': df_test['id'], 'sentiment': test_label})
    return submission

def submit(submission_df):
    submission_df.to_csv("output/submission.csv", index=False)

def test(df_train, df_test):
    cleaned_train_reviews = []
    cleaned_test_reviews = []

    for i in range(df_train.index.size):
        cleaned_train_reviews.append(clean_review(df_train.loc[i, 'review']))

    for i in range(df_test.index.size):
        cleaned_test_reviews.append(clean_review(df_test.loc[i, 'review']))

    # 将训练集和测试集都作为语料库，
    corpus = cleaned_train_reviews + cleaned_test_reviews

    vectorizer = CountVectorizer(lowercase=None, ngram_range=(1, 2), stop_words=None, max_features=5000)
    # vectorizer = TfidfVectorizer(lowercase=None, ngram_range=(1, 1), stop_words=None, max_features=5000)

    vectorizer.fit(corpus)

    train_features = vectorizer.transform(cleaned_train_reviews)
    test_features = vectorizer.transform(cleaned_test_reviews)

    train_features = train_features.toarray()
    test_features = test_features.toarray()

    # clf = RandomForestClassifier()
    clf = LogisticRegression()
    clf = clf.fit(train_features, df_train['sentiment'])

    test_label = clf.predict(test_features)
    print("Score:" + str(clf.score(train_features, df_train['sentiment'])))
    submission = pd.DataFrame(data={'id': df_test['id'], 'sentiment': test_label})
    return submission


if __name__ == '__main__':
    train_df = pd.read_csv("data/labeledTrainData.tsv", sep="\t", encoding="latin-1")
    test_df = pd.read_csv("data/testData.tsv", sep="\t", encoding="latin-1")

    submission = test(train_df, test_df)
    submit(submission)
    print(submission.head())

