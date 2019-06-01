# -*- UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def learn_lstm():
    # 数据向量维数input_size=3, 隐藏元维度 hidden_size = 3, num_layers = 1 个 LSTM 层串联(如果是1，可以省略，默认为1)
    lstm = nn.LSTM(3, 3)

    sequence1 = [torch.randn(1, 3) for _ in range(5)]  # a list of tensor, 使用torch.cat进行将tensor串成一个tensor
    sequence2 = [torch.randn(1, 3) for _ in range(5)]
    sequence3 = [torch.randn(1, 3) for _ in range(5)]
    # (num_layers * num_directions, batch, hidden_size)
    hidden0 = torch.randn(1, 1, 3)
    cell0 = torch.randn(1, 1, 3)
    # input shape : (seq_len, batch, input_size)
    # seq_len 序列长度，因为句子是变长的，所以一般都会 pandding,
    # torch.nn.utils.rnn.pack_padded_sequence()以及torch.nn.utils.rnn.pad_packed_sequence()处理 padding
    inputs = torch.cat(sequence1).view(5, 1, 3)
    # output shape: (seq_len, batch, num_directions * hidden_size):
    output, (hidden, cell) = lstm(inputs, (hidden0, cell0))
    print(sequence1.size())
    print(output.size())
    print(hidden)


# An LSTM for Part-of-Speech Tagging 词性标注

# prepare data
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# model
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)  # 最后的 Softmax 层的linear
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 实际上此处的 hidden 包括 h 和 c
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embedding = self.word_embedding(sentence)
        # 注意此处 len(sentence) 是变长的
        lstm_out, self.hidden = self.lstm(embedding.view(len(sentence), 1, -1), self.hidden)
        # lstm output shape shape: (seq_len, batch, num_directions * hidden_size)，此处-1 就是hidden_dim
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train(training_data, word_to_ix, tag_to_ix):
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[1][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)

    for epoch in range(100):
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[1][0], word_to_ix)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)

    return model


def infer(model, sentence_str):
    with torch.no_grad():
        inputs = prepare_sequence(sentence_str.split(), word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)


if __name__ == '__main__':
    training_data = [
        ("The dog ate the apple pie".split(), ["DET", "NN", "V", "DET", "NN", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 3
    # print(training_data)
    model = train(training_data, word_to_ix, tag_to_ix)
    sentence_str = "The book that the apple"
    infer(model, sentence_str)
