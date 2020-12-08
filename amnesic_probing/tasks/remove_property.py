"""
Usage:
  remove_property.py [--vecs=VECS] [--labels=LABELS] [--out_dir=OUT_DIR]
        [--n_cls=N_CLS] [--task=TASK] [--input_dim=INPUT_DIM] [--max_iter=MAX_ITER]
        [--balance_data=BALANCE_DATA] [--n=N]
        [--wandb]

Options:
  -h --help                     show this help message and exit
  --vecs=VECS                   input vectors file. using the train path (and automatically also using the dev,
                                by replacing train by dev)
  --labels=LABELS               labels file. using the train path (and automatically also using the dev,
                                by replacing train by dev)
  --out_dir=OUT_DIR             logs and outputs directory
  --n_cls=N_CLS                 number of classifiers to use [default: 20]
  --task=TASK                   task type. between word_ind, sen_len, task [default: task]
  --input_dim=INPUT_DIM         input dimension [default: 768]
  --max_iter=MAX_ITER           maximum iteration for the linear model [default: 10000]
  --balance_data=BALANCE_DATA   balance data based on the labels. [default: false]
  --n=N                         number of training examples [default: 100000]
  --wandb                       log using wandb

"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import wandb
from docopt import docopt
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.utils import read_files, get_projection_matrix, get_regression_pls, classification_tasks

np.random.seed(0)


def balance_data(x, y):
        df = pd.DataFrame.from_dict({'x': x.tolist(), 'y': y})
        g = df.groupby('y')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
        return np.array(g['x'].tolist()), np.array(g['y'].tolist())


def log_wandb(arguments):
    labels = arguments['--labels'].split('.')[0]
    task_type = labels.split('/')[-1]

    task = arguments['--task']
    if task != 'task':
        task_type = task
    if task in ['word_ind', 'sen_len']:
        classification_type = 'regression'
    else:
        classification_type = 'classification'

    data_orig = labels.split('data/')[1].split('/')[0]
    print(labels)
    print(data_orig)
    dataset = data_orig.split('_output', 1)[0]
    masking = data_orig.rsplit('_', 1)[1]

    layer_str = arguments['--vecs'].split('/')[-1].rsplit('.', maxsplit=1)[0]
    if 'layer' in layer_str:
        layer = str(layer_str.split(':')[1])
    else:
        layer = 'last'

    config = dict(
        property=task_type,
        encoder='bert-base-uncased',
        dataset=dataset,
        masking=masking,
        layer=layer
    )

    wandb.init(
        name=task_type + f'_{layer}_inlp',
        project="amnesic_probing",
        tags=["inlp", task_type, classification_type],
        config=config,
    )


if __name__ == '__main__':
    arguments = docopt(__doc__)

    if arguments['--wandb']:
        log_wandb(arguments)

    out_dir = arguments['--out_dir']

    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(out_dir + '/P.npy'):
        print('matrix already exists... skipping training')
        exit(0)
    writer = SummaryWriter(out_dir)

    in_dim = int(arguments['--input_dim'])

    sentence_file = arguments['--vecs'].rsplit('/', 1)[0] + '/' + 'tokens.pickle'

    vecs_train, labels_train, sentences_train = read_files(arguments['--vecs'], arguments['--labels'],
                                                           sentence_file,
                                                           ignore_special_tokens=True)
    vecs_dev, labels_dev, sentences_dev = read_files(arguments['--vecs'].replace('train', 'dev'),
                                                     arguments['--labels'].replace('train', 'dev'),
                                                     sentence_file.replace('train', 'dev'),
                                                     ignore_special_tokens=True)

    print('#sentences', len(vecs_train))

    task = arguments['--task']

    (x_train, y_train, _), (x_dev, y_dev, _) = get_appropriate_data(task, vecs_train, labels_train, sentences_train,
                                                                    vecs_dev, labels_dev, sentences_dev)

    if bool(arguments['--balance_data'] == 'true'):
        x_train, y_train = balance_data(x_train, y_train)
        x_dev, y_dev = balance_data(x_dev, y_dev)

    n_classes = len(set(y_train))
    majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))
    print('number of classes:', n_classes)
    print('most common class (dev):', majority)

    if arguments['--wandb']:
        wandb.run.summary['n_classes'] = n_classes
        wandb.run.summary['majority'] = majority

    num_clfs = int(arguments['--n_cls'])
    max_iter = int(arguments['--max_iter'])

    # setting n to be the minimum between the number of examples
    # in the training data and the provided amount
    n = min(int(arguments['--n']), len(y_train))
    print('using {} training examples'.format(n))

    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=n)
    if task in classification_tasks:
        P, all_projections, best_projection = get_projection_matrix(num_clfs,
                                                                    x_train, y_train, x_dev, y_dev,
                                                                    majority_acc=majority, max_iter=max_iter,
                                                                    summary_writer=writer)
    else:
        P, all_projections, best_projection = get_regression_pls(num_clfs, x_train,
                                                                 y_train, x_dev, y_dev, dim=in_dim,
                                                                 majority_acc=majority,
                                                                 summary_writer=writer)

    for i, projection in enumerate(all_projections):
        np.save(out_dir + '/P_{}.npy'.format(i), projection)

    np.save(out_dir + '/P.npy', best_projection[0])

    if task in classification_tasks:
        removed_directions = int((best_projection[1]) * n_classes)
        # in case of 2 classes, each inlp iteration we remove a single direction
        if n_classes == 2:
            removed_directions /= 2
    else:  # in regression tasks, each iteration we remove a single dimension
        removed_directions = int((best_projection[1]))

    meta_dic = {'best_i': best_projection[1],
                'n_classes': n_classes,
                'majority': majority,
                'removed_directions': removed_directions}

    if arguments['--wandb']:
        wandb.run.summary['best_i'] = best_projection[1]
        wandb.run.summary['removed_directions'] = removed_directions

    json.dump(meta_dic, open(out_dir + '/meta.json', 'w'))

    print('done iterations. exiting...')
