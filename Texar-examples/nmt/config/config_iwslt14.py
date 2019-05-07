
num_epochs = 15
# every 500 step display
display = 500

source_vocab_file = './data/iwslt14/vocab.de'
target_vocab_file = './data/iwslt14/vocab.en'

train = {
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14/train.de',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/iwslt14/train.en',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    }
}

val = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/valid.en',
        'vocab_file': target_vocab_file,
    }
}

test = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/test.en',
        'vocab_file': target_vocab_file,
    }
}

debug_val = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/debug.valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/debug.valid.en',
        'vocab_file': target_vocab_file,
    }
}

debug_test = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/debug.test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/debug.test.en',
        'vocab_file': target_vocab_file,
    }
}