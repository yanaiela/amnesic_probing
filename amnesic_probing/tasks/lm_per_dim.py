"""
Usage:
  lm.py [--vecs=VECS] [--labels=LABELS] [--text=TEXT]
        [--task=TASK]
        [--deprobe_dir=DEPROBE_DIR]
        [--display_examples=DISPLAY_EXAMPLES]
        [--device=DEVICE]
        [--n=N]
        [--wandb]

Options:
  -h --help                     show this help message and exit
  --vecs=VECS                   input vectors file
  --labels=LABELS               labels file
  --text=TEXT                   text file
  --task=TASK                   task type. between word_ind, sen_len, task [default: task]
  --deprobe_dir=DEPROBE_DIR     directory where the amnesic_probing files are located.
  --display_examples=DISPLAY_EXAMPLES       number of examples to display [default: 10]
  --device=DEVICE               cpu, cuda:0, cuda:1, ... [default: cpu]
  --n=N                         number of training examples [default: 100000]
  --wandb                       log using wandb

"""

import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import wandb
from docopt import docopt
from sklearn.utils import shuffle

from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.lm import eval_lm_performance, eval_task_performance
from amnesic_probing.tasks.utils import data_projection, read_files, get_lm_vals

_, tokenizer, out_embed, bias = get_lm_vals('bert-base-uncased')


def log_wandb(arguments):
    task_name = arguments['--deprobe_dir'].split('models/lm/')[1]
    task_type = task_name.split('/')[0]
    layer = arguments['--vecs'].split('/')[-1].split('.')[0]

    labels = arguments['--labels'].split('.')[0]
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
        layer=layer
    )

    wandb.init(
        name=task_type + '_eval_iterations',
        project="amnesic_probing",
        tags=["lm", "eval", task_type],
        config=config,
    )


def load_deprobing_params(in_file):
    with open(in_file, 'r') as f:
        meta = json.load(f)
    return meta


if __name__ == '__main__':
    arguments = docopt(__doc__)

    deprobe_dir = arguments['--deprobe_dir']
    if not os.path.isdir(deprobe_dir):
        assert 'Deprobing directory does not exists...'

    use_wandb = arguments['--wandb']
    if use_wandb:
        log_wandb(arguments)

    vecs_train, labels_train, sentences_train = read_files(arguments['--vecs'],
                                                           arguments['--labels'],
                                                           arguments['--text'], ignore_special_tokens=True)
    vecs_dev, labels_dev, sentences_dev = read_files(arguments['--vecs'].replace('train', 'dev'),
                                                     arguments['--labels'].replace('train', 'dev'),
                                                     arguments['--text'].replace('train', 'dev'),
                                                     ignore_special_tokens=True)

    task = arguments['--task']

    (x_train, y_train, words_train), (x_dev, y_dev, words_dev) = get_appropriate_data(task, vecs_train, labels_train,
                                                                                      sentences_train,
                                                                                      vecs_dev, labels_dev,
                                                                                      sentences_dev)

    pos2ind = {p: i for i, p in enumerate(sorted(set([item for sublist in labels_train for item in sublist])))}

    print('number of classes', len(pos2ind))
    print('most common class', Counter(y_dev).most_common(1)[0][1] / float(len(y_dev)))

    meta = load_deprobing_params(deprobe_dir + '/meta.json')
    n_coords = int(meta['removed_directions'])
    n_classes = len(pos2ind)

    if use_wandb:
        wandb.run.summary['n_classes'] = len(pos2ind)
        wandb.run.summary['majority'] = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))
        wandb.run.summary['removed_directions'] = n_coords

    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=min(len(y_train), int(arguments['--n'])))
    device = arguments['--device']

    results_dic = {}
    for proj_file in glob.glob(deprobe_dir + '/P_*.npy'):
        P = np.load(proj_file)

        print('evaluating performance')
        proj_iter = int(proj_file.split('/P_')[1].split('.npy')[0])

        removed_directions = int((proj_iter + 1) * n_classes)
        # in case of 2 classes, each inlp iteration we remove a single direction
        if n_classes == 2:
            removed_directions /= 2

        print(removed_directions)
        lm_results = eval_lm_performance(tokenizer, out_embed, bias, x_dev, words_dev, P,
                                         n_coords=int(removed_directions), device=device)

        print('removing property from inputs')

        x_train_no_label = data_projection(x_train, P)
        x_dev_no_label = data_projection(x_dev, P)

        task_results = eval_task_performance(x_train, y_train, x_dev, y_dev, x_train_no_label, x_dev_no_label, task)

        all_results = {**lm_results, **task_results}
        for k, v in all_results.items():
            print(k, v)

        results_dic[proj_iter] = all_results

    results_df = pd.DataFrame(results_dic).T
    results_df.index = results_df.index.set_names('iter')
    results_df.columns = results_df.columns
    results_df = results_df.reset_index()

    if use_wandb:
        table_data = results_df.values.tolist()
        wandb.log({"results": wandb.Table(data=table_data, columns=results_df.columns.tolist())})
