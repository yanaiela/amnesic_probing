"""
Usage:
  remove_property.py [--labels=LABELS] [--out_labels=OUT_LABELS] [--keep_label=KEEP_LABEL]

Options:
  -h --help                     show this help message and exit
  --labels=LABELS               labels file. using the train path (and automatically also using the dev,
                                by replacing train by dev)
  --out_labels=OUT_LABELS       output file for the new labels
  --keep_label=KEEP_LABEL       name of the label to keep

"""

import pickle

from docopt import docopt
from tqdm import tqdm


def read_data(in_f):
    with open(in_f, 'rb') as f:
        labels = pickle.load(f)
    return labels


def convert_labels(in_f, label2keep):
    labels = read_data(in_f)

    keep_labels = label2keep.split(',')

    new_labels = []
    for sen_labels in tqdm(labels):
        sen_new_labels = []
        for l in sen_labels:
            if l in keep_labels:
                sen_new_labels.append(l)
            else:
                sen_new_labels.append('other')
        new_labels.append(sen_new_labels)
    return new_labels


def to_file(labels, out_f):
    with open(out_f, 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    labels_file = arguments['--labels']

    new_labels = convert_labels(labels_file, arguments['--keep_label'])

    to_file(new_labels, arguments['--out_labels'])
