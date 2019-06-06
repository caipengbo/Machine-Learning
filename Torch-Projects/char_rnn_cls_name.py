# -*- UTF-8 -*-
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math

# 姓氏分类（data/names包含各个国家的姓氏）

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of names per language {language: [names ...]}
def build_category_dict():
    category_dict = {}
    category_list = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        category_list.append(category)
        lines = readLines(filename)
        category_dict[category] = lines

    return category_dict, category_list


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor (one-hot encode)
def letter2tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.find(letter)] = 1
    return tensor


def name2tensor(name):
    # all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    # Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_vec, hidden):
        combined = torch.cat((input_vec, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def category_from_output(output, category_list):
    # 选择 softmax 中最大的概率值，看是哪个 category
    top_number, top_index = output.topk(1)
    category_i = top_index[0].item()
    return category_list[category_i], category_i


def random_chose(l):
    return l[random.randint(0, len(l) - 1)]


def random_create_training_example(category_dict, category_list):
    all_letters = string.ascii_letters + " .,;'"
    category = random_chose(category_list)
    name = random_chose(category_dict[category])
    category_tensor = torch.tensor([category_list.index(category)], dtype=torch.long)
    name_tensor = name2tensor(name)
    return category, name, category_tensor, name_tensor


def train_step(category_tensor, name_tensor, rnn_model, criterion, optimizer):
    # Create a zeroed initial hidden state
    hidden = rnn_model.initHidden()
    rnn_model.zero_grad()

    # Read each letter in and Keep hidden state for next letter
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn_model(name_tensor[i], hidden)

    # Compare final output to target
    loss = criterion(output, category_tensor)

    # Back-propagate
    loss.backward()
    # Update parameters
    optimizer.step()
    # Return the output and loss
    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(category_dict, category_list, rnn_model):
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(rnn_model.parameters(), lr=0.005)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, name_tensor = random_create_training_example(category_dict, category_list)
        output, loss = train_step(category_tensor, name_tensor, rnn_model, criterion, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, category_list)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


# Just return an output given a line
def evaluate(name_tensor, rnn_model):
    hidden = rnn_model.initHidden()

    for i in range(name_tensor.size()[0]):
        output, hidden = rnn_model(name_tensor[i], hidden)

    return output


def predict(rnn_model, name, category_list, n_predictions=3):
    output = evaluate(name2tensor(name), rnn_model)

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, category_list[category_index]))
        predictions.append([value, category_list[category_index]])

    return predictions


if __name__ == '__main__':
    # category_dict:{'language':[name1, name2], }
    category_dict, category_list = build_category_dict()
    n_categories = len(category_list)  # n 类 language, dict 的 key
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    # print(category_list)
    # print(category_dict['Chinese'][:15])
    # ====================Training==========================
    # train(category_dict, category_list, rnn)
    # torch.save(rnn, 'saved_model/char-rnn-cls-name.pt')
    rnn = torch.load('saved_model/char-rnn-cls-name.pt')
    name = 'Akita'
    predict(rnn, name, category_list, n_predictions=3)