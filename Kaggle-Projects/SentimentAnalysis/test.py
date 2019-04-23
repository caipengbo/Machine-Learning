# -*- UTF-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def test():
    list = [
        'This is the first document.',
        'This is is the second document.'
    ]
    list2 = [
        'I am from list2.',
        'Is this the first document?'
    ]
    vectorizer = CountVectorizer()
    vectorizer.fit(list, list2)
    print(vectorizer.vocabulary_)

    listall = list + list2
    vectorizer.fit(list+list2)
    print(vectorizer.vocabulary_)

    x = vectorizer.transform(list)
    print(x.toarray())

def test2():
    list1 = [1, 2, 3, 4]
    list2 = [11, 12, 13, 14]
    list1.append(list2)
    print(list1)


def test3():
    df5000 = pd.read_csv("data/testData.tsv", sep="\t", encoding="latin-1")
    df25000 = pd.read_csv("data/labeledTrainData25000.tsv", sep="\t", encoding="latin-1")
    df5000 = df5000.drop(['review'], axis=1)

    list = []

    for r in df5000['id']:
        list += df25000.loc[df25000['id'] == r]['sentiment'].values.tolist()

    df5000['sentiment'] = list

    # for i in :
    #     id = df5000.loc[i, 'id']
    #     df5000.loc[i, 'sentiment'] = df25000.loc[df25000['id'] == id, ['sentiment']]



    df5000.to_csv("data/answer.csv", index=False)

def test4():
    df = pd.DataFrame({"col1": [1, 2, 3], 'col2': ['1', '2', '3']}, dtype='int32')
    # df['col1'] = df['col1'].astype('str')
    # df = df.astype({'col1': 'str', 'col2': 'int32'})

    print(df)
    print(df.loc[:,'col1':'col2'])

if __name__ == '__main__':
    test4()