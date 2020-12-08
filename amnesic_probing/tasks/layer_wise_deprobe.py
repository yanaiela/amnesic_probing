"""
Usage:
  layer_wise_deprobe.py [--layers=LAYERS] [--proj_vecs=PROJ_VECS] [--labels=LABELS] [--text=TEXT] [--task=TASK]
        [--n=N]
        [--wandb]

Options:
  -h --help                     show this help message and exit
  --layers=LAYERS               input folder of the plain layer vectors
  --proj_vecs=proj_vecs         input dir of all the projected layers
  --labels=LABELS               labels file. using the train path (and automatically also using the dev,
                                by replacing train by dev)
  --text=TEXT                   text file
  --task=TASK                   task type. between word_ind, sen_len, task [default: task]
  --n=N                         number of training examples [default: 100000]
  --wandb                       log using wandb

"""

import pickle

import numpy as np
import wandb
from docopt import docopt
from sklearn.utils import shuffle
from tqdm import tqdm

from amnesic_probing.tasks.data_preparation import create_labeled_data
from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.utils import learn_cls, read_files, learn_pls_cls, classification_tasks


def get_labels_file(in_dir):
    with open(in_dir, 'rb') as f:
        labels = pickle.load(f)
    return labels


def get_layer_vecs(layer_file):
    vectors = np.load(layer_file, allow_pickle=True)
    vectors = np.array([x[1:-1] for x in vectors])
    return vectors


def learn_cls_for_layer(layer_name, labels_train, labels_dev, n):
    vecs_train = get_layer_vecs(layer_name)
    vecs_dev = get_layer_vecs(layer_name.replace('train', 'dev'))
    x_train, y_train, label2i = create_labeled_data(vecs_train, labels_train)
    x_dev, y_dev, _ = create_labeled_data(vecs_dev, labels_dev, pos2i=label2i)
    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=n)

    score = learn_cls(x_train, y_train, x_dev, y_dev)
    return score


def learn_cls_for_layer_new(layer_name, labels_name, text_name, task, n):
    vecs_train, labels_train, sentences_train = read_files(layer_name,
                                                           labels_name,
                                                           text_name, ignore_special_tokens=True)
    vecs_dev, labels_dev, sentences_dev = read_files(layer_name.replace('train', 'dev'),
                                                     labels_name.replace('train', 'dev'),
                                                     text_name.replace('train', 'dev'),
                                                     ignore_special_tokens=True)

    (x_train, y_train, words_train), (x_dev, y_dev, words_dev) = get_appropriate_data(task, vecs_train, labels_train,
                                                                                      sentences_train,
                                                                                      vecs_dev, labels_dev,
                                                                                      sentences_dev)
    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=n)
    if task in classification_tasks:
        score = learn_cls(x_train, y_train, x_dev, y_dev)
    else:
        score = learn_pls_cls(x_train, y_train, x_dev, y_dev)
    return score


def compute_per_layer_task_performance(in_dir, labels_file, text_file, task, n):
    print('layer probing for tasks')

    per_layer_score = np.zeros(14)

    for layer_index in tqdm(range(0, 14)):
        layer_name = f'{in_dir}/vec_layer:{layer_index}.npy'
        if layer_index == 13:
            layer_name = f'{in_dir}/last_vec.npy'

        score = learn_cls_for_layer_new(layer_name, labels_file, text_file, task, n)
        per_layer_score[layer_index] = score

        print(layer_index, score)

    return per_layer_score


def compute_following_layers_task_performance(in_dir, labels_file, text_file, task, n):
    print('layer amnesic_probing. removing frmo layer i and testing on layer j')

    layers_task_results = []

    for from_layer in range(0, 13):
        for to_layer in range(from_layer + 1, 14):
            vecs_name = f'{in_dir}/from:{from_layer}.to:{to_layer}.npy'
            score = learn_cls_for_layer_new(vecs_name, labels_file, text_file, task, n)

            layers_task_results.append([from_layer, to_layer, score])

            print(from_layer, to_layer, score)
    return layers_task_results


def log_wandb(arguments):
    labels = arguments['--labels'].split('.')[0]
    task_type = labels.split('/')[-1]

    task = arguments['--task']
    if task != 'task':
        task_type = task
        classification_type = 'regression'
    else:
        classification_type = 'classification'

    data_orig = labels.split('data/')[1].split('/')[0]
    print(labels)
    print(data_orig)
    dataset = data_orig.split('_output', 1)[0]
    masking = data_orig.rsplit('_', 1)[1]

    config = dict(
        property=task_type,
        encoder='bert-base-uncased',
        dataset=dataset,
        masking=masking,
    )

    wandb.init(
        name=f'{task_type}_layer_wise_eval',
        project="amnesic_probing",
        tags=["layer_wise", task_type, classification_type],
        config=config,
    )


if __name__ == '__main__':
    arguments = docopt(__doc__)

    labels = arguments['--labels']
    layers = arguments['--layers']
    texts = arguments['--text']
    task = arguments['--task']
    n = int(arguments['--n'])
    proj_vecs = arguments['--proj_vecs']

    use_wandb = arguments['--wandb']
    if use_wandb:
        log_wandb(arguments)

    layer_probe_results = compute_per_layer_task_performance(layers, labels, texts, task, n)
    layers_task_results = compute_following_layers_task_performance(proj_vecs, labels, texts, task, n)

    if use_wandb:
        layer_probing = np.concatenate((np.array(list(range(14))).reshape(14, 1),
                                        layer_probe_results.reshape(14, 1)),
                                       axis=1)
        wandb.log({"layer_probe": wandb.Table(data=layer_probing.tolist(),
                                              columns=['layer', 'probing'])})

        wandb.log({'per_layer_deprobe': wandb.Table(data=layers_task_results,
                                                    columns=['from', 'to', 'score'])})
