# coding=utf-8
import sys

from drqa.retriever import ElasticDocRanker

sys.path.extend(['/home/LAB/caipb/Machine-Learning-Deep-Learning/DrQA'])

if __name__ == '__main__':
    # es = ElasticDocRanker(elastic_url='10.1.1.9:9266', elastic_index='test3', )
    d = {'question': {'Title': 'question title', 'Body': 'Body content'}, 'answer': 'answer content'}

    idx = d.copy()
    print(idx['question'])
    for field in ['question', 'Title']:
        idx = idx[field]
    print(idx)
