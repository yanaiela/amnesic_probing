"""
Usage:
  task_specific_eval.py [--vecs=VECS] [--labels=LABELS] [--text=TEXT]
        [--task=TASK]
        [--deprobe_dir=DEPROBE_DIR]
        [--device=DEVICE]
        [--wandb]

Options:
  -h --help                     show this help message and exit
  --vecs=VECS                   input vectors file
  --labels=LABELS               labels file
  --text=TEXT                   text file
  --task=TASK                   task type. between word_ind, sen_len, task [default: task]
  --deprobe_dir=DEPROBE_DIR     directory where the amnesic_probing files are located.
  --device=DEVICE               cpu, cuda:0, cuda:1, ... [default: cpu]
  --wandb                       log using wandb

"""

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import wandb
from docopt import docopt

from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.lm import load_deprobing_params
from amnesic_probing.tasks.utils import data_projection, read_files, get_lm_vals, get_lm_predictions_gpu, \
    rand_direction_control

_, tokenizer, out_embed, bias = get_lm_vals('bert-base-uncased')


def eval_lm_per_task(tokenizer, out_embed, bias, x, words, y, projection, n_coords, label2ind, device='cpu'):
    y_ids = np.array(tokenizer.convert_tokens_to_ids(words))

    lm_results = defaultdict(dict)

    for label_name, label_ind in label2ind.items():
        labels_indices = y == label_ind
        if sum(labels_indices) == 0:
            continue
        x_labels = x[labels_indices]
        y_id_labels = y_ids[labels_indices]

        base_acc = get_lm_predictions_gpu(out_embed, bias, x_labels, y_id_labels, device=device)

        x_p = data_projection(x_labels, projection)
        p_acc = get_lm_predictions_gpu(out_embed, bias, x_p, y_id_labels, device=device)

        x_rand_dir = rand_direction_control(x_labels, n_coords)
        rand_dir_acc = get_lm_predictions_gpu(out_embed, bias, x_rand_dir, y_id_labels, device=device)

        lm_results[label_name]['vanilla'] = base_acc
        lm_results[label_name]['p'] = p_acc
        lm_results[label_name]['rand_dir'] = rand_dir_acc

    return pd.DataFrame(lm_results).T


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
        name=task_type + '_task_specific_eval',
        project="amnesic_probing",
        tags=["lm", "eval", "task_specific", task_type],
        config=config,
    )


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

    if use_wandb:
        wandb.run.summary['n_classes'] = len(pos2ind)
        wandb.run.summary['majority'] = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))

    proj_file = deprobe_dir + '/P.npy'
    meta = load_deprobing_params(deprobe_dir + '/meta.json')
    n_coords = int(meta['removed_directions'])

    if os.path.isfile(proj_file):
        P = np.load(proj_file)
    else:
        raise FileNotFoundError('projection file does not exists...')

    print('evaluating performance')
    # calculating the number of dimensions that were removed

    device = arguments['--device']
    lm_results_df = eval_lm_per_task(tokenizer, out_embed, bias, x_dev, words_dev, y_dev, P, n_coords, pos2ind,
                                     device=device)

    if use_wandb:
        labels_names = lm_results_df.index.to_numpy()
        table_results = np.concatenate((labels_names.reshape(len(labels_names), 1), lm_results_df.to_numpy()), axis=1).tolist()
        wandb.log({"results": wandb.Table(data=table_results,
                                          columns=['label'] + lm_results_df.columns.tolist())})
