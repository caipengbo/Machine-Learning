# -*- UTF-8 -*-
import random

import torch

from nmt.model import EncoderRNN, AttnDecoderRNN
from nmt.prepare import prepareData
from nmt.train_evaluation import evaluateRandomly, trainIters

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_lang, output_lang, pairs = prepareData('english', 'french', True)
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    encoder1, attn_decoder1 = trainIters(encoder1, attn_decoder1, 75000, device, print_every=5000)

    torch.save(encoder1, "saved_model/seq2seq-nmt-encoder.pt")
    torch.save(attn_decoder1, "saved_model/seq2seq-nmt-decoder.pt")

    # evaluateRandomly(encoder1, attn_decoder1)
