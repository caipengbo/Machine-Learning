# -*- UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 词袋模型的逻辑回归（辨别西班牙语和英语）
# 简单的前馈神经网络


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


class BoWClassifier(nn.Module):
    def __init__(self, vacab_size, num_labels):
        super(BoWClassifier, self).__init__()
        # affine map
        self.linear = nn.Linear(in_features=vacab_size, out_features=num_labels)

    def forward(self, bow_vec):
        # non-linearities
        return F.log_softmax(self.linear(bow_vec), dim=1)

if __name__ == '__main__':
    train_data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
            ("Give it to me".split(), "ENGLISH"),
            ("No creo que sea una buena idea".split(), "SPANISH"),
            ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

    test_data = [("Yo creo que si".split(), "SPANISH"),
                 ("it is lost on me".split(), "ENGLISH")]

    # 将 word 转换成 索引
    word_to_ix = {}
    for sent, _ in train_data + test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix)

    VOCAB_SIZE = len(word_to_ix) # 词袋向量的维度
    NUM_LABELS = 2

    label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

    model = BoWClassifier(VOCAB_SIZE, NUM_LABELS)

    # 模型的参数（nn.Linear() 中的 A 和 b）
    for parm in model.parameters():
        print(parm.size())

    # 使用模型进行推理（运行一遍模型），这个时候不需要训练，所以将代码 wrapped in torch.no_grad()
    with torch.no_grad():
        sample = train_data[0]
        bow_vector = make_bow_vector(sample[0], word_to_ix)
        log_probs = model(bow_vector)
        print(log_probs)

    with torch.no_grad():
        for instance, label in test_data:
            bow_vec = make_bow_vector(instance, word_to_ix)
            log_probs = model(bow_vec)
            print(log_probs)

    # 训练前的参数
    print("=====训练前=====")
    for param in model.parameters():
        print(param)

    # 训练
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for instance, label in train_data:
            model.zero_grad()
            bow_vec = make_bow_vector(instance, word_to_ix)
            target = make_target(label, label_to_ix)
            log_probs = model(bow_vec)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

    # 训练后的参数
    print("=====训练后=====")
    for param in model.parameters():
        print(param)

    with torch.no_grad():
        for instance, label in test_data:
            bow_vec = make_bow_vector(instance, word_to_ix)
            log_probs = model(bow_vec)
            print(log_probs)