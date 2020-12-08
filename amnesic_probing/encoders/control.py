"""
Usage:
  encode.py [--input_sentences=INPUT_SENTENCES] [--input_labels=INPUT_LABELS] [--output_file=OUTPUT_FILE]

Options:
  -h --help                     show this help message and exit
  --input_sentences=INPUT_SENTENCES       input file containing sentences. pickle file
  --input_labels=INPUT_LABELS   input file containing labels. pickle file
  --output_file=OUTPUT_FILE     output file where to write the output file

"""

import pickle

import numpy as np
from docopt import docopt

np.random.seed(0)


def get_num_labels(in_f):
    with open(in_f, 'rb') as f:
        labels = pickle.load(f)

    all_labels = [x for l in labels for x in l]
    return len(set(all_labels))


def assign_labels(in_f, n_labels):
    with open(in_f, 'rb') as f:
        lines = pickle.load(f)

    words = [x for l in lines for x in l]
    words = list(set(words))
    np.random.shuffle(words)

    temp = dict(enumerate(words))
    temp = {v: k for k, v in temp.items()}
    random_words_label_dic = {}
    for k, v in temp.items():
        random_words_label_dic[k] = v % n_labels
    return random_words_label_dic


def label_sentences(in_f, words_labels_dic, n_labels):
    with open(in_f, 'rb') as f:
        lines = pickle.load(f)

    labels = []
    for line in lines:
        sentence_labels = []
        for w in line:
            if w not in words_labels_dic:
                i = np.random.randint(n_labels)
                words_labels_dic[w] = i
            sentence_labels.append(words_labels_dic[w])
        labels.append(sentence_labels)

    return labels, words_labels_dic


if __name__ == '__main__':
    arguments = docopt(__doc__)

    n = get_num_labels(arguments['--input_labels'])
    words_labels = assign_labels(arguments['--input_sentences'], n)

    train_labels, words_labels = label_sentences(arguments['--input_sentences'], words_labels, n)
    dev_labels, words_labels = label_sentences(arguments['--input_sentences'].replace('train', 'dev'), words_labels, n)

    out_file = arguments['--output_file']
    # os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(train_labels, f)

    out_file_dev = out_file.replace('train', 'dev')
    # os.makedirs(out_file_dev, exist_ok=True)
    with open(out_file_dev, 'wb') as f:
        pickle.dump(dev_labels, f)

