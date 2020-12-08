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

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import wandb
from docopt import docopt
from sklearn.utils import shuffle

from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.utils import data_projection, read_files, get_lm_vals, dropout_control, rand_direction_control, \
    get_lm_predictions_gpu, dkl_gpu, learn_cls, learn_reg_cls, get_lm_predictions, calc_entropy, most_probable_label, \
    classification_tasks

_, tokenizer, out_embed, bias = get_lm_vals('bert-base-uncased')


def unify_inputs(sentences, labels_seq, vecs, pos2ind):
    x = []
    y = []
    words = []

    for sen, label, vec in zip(sentences, labels_seq, vecs):
        for w, l, v in zip(sen, label, vec):
            x.append(v)
            y.append(pos2ind[l])
            words.append(w)

    return np.array(x), np.array(y), words


def sentence_prediction_example(text_tokens, text_vecs, task_labels, projection_matrix):
    print('sentence: ', ' '.join(text_tokens))
    print('token', 'lm predicted token', 'lm task-less predicted token', 'task label')
    outputs = []
    predicted_tokens = get_lm_predictions(tokenizer, out_embed, bias, text_vecs)
    predicted_tokens_p = get_lm_predictions(tokenizer, out_embed, bias,
                                            data_projection(text_vecs, projection_matrix))
    for true_word, y_hat, y_hat_p, y_task in zip(text_tokens, predicted_tokens, predicted_tokens_p, task_labels):
        print(true_word, y_hat, y_hat_p, y_task)
        outputs.append([true_word, y_hat, y_hat_p, y_task])
    return outputs


def eval_lm_performance(tokenizer, out_embed, bias, x, words, projection, n_coords, device='cpu'):
    y_ids = tokenizer.convert_tokens_to_ids(words)

    lm_results = {}
    base_acc = get_lm_predictions_gpu(out_embed, bias, x, y_ids, device=device)

    x_p = data_projection(x, projection)
    p_acc = get_lm_predictions_gpu(out_embed, bias, x_p, y_ids, device=device)

    x_dropout = dropout_control(x, n_coords)
    dropout_acc = get_lm_predictions_gpu(out_embed, bias, x_dropout, y_ids, device=device)

    x_rand_dir = rand_direction_control(x, n_coords)
    rand_dir_acc = get_lm_predictions_gpu(out_embed, bias, x_rand_dir, y_ids, device=device)

    lm_results['lm_acc_vanilla'] = base_acc
    lm_results['lm_acc_p'] = p_acc
    lm_results['lm_acc_dropout'] = dropout_acc
    lm_results['lm_acc_rand_dir'] = rand_dir_acc

    dkl_p, x_probs = dkl_gpu(out_embed, bias, x, x_p, y_ids, device=device)
    dkl_drop, _ = dkl_gpu(out_embed, bias, x, x_dropout, y_ids, x_probs, device=device)
    dkl_rand, _ = dkl_gpu(out_embed, bias, x, x_rand_dir, y_ids, x_probs, device=device)
    lm_results['dkl_p'] = dkl_p
    lm_results['dkl_dropout'] = dkl_drop
    lm_results['dkl_rand_dir'] = dkl_rand

    return lm_results


def eval_topk_performance(tokenizer, out_embed, bias, x, words, projection, probable_labels,
                          n_coords, y_train_labels, k=100, device='cpu'):
    y_ids = tokenizer.convert_tokens_to_ids(words)

    lm_labels_results = {}
    entropy_vanilla = calc_entropy(x, y_ids, y_train_labels, probable_labels, tokenizer, out_embed, bias, k, device)

    x_p = data_projection(x, projection)
    entropy_p = calc_entropy(x_p, y_ids, y_train_labels, probable_labels, tokenizer, out_embed, bias, k, device)

    x_dropout = dropout_control(x, n_coords)
    entropy_dropout = calc_entropy(x_dropout, y_ids, y_train_labels, probable_labels, tokenizer, out_embed, bias, k, device)

    x_rand_dir = rand_direction_control(x, n_coords)
    entropy_rand_dir = calc_entropy(x_rand_dir, y_ids, y_train_labels, probable_labels, tokenizer, out_embed, bias, k, device)

    lm_labels_results['top_k_entropy_vanilla'] = entropy_vanilla
    lm_labels_results['top_k_entropy_p'] = entropy_p
    lm_labels_results['top_k_entropy_dropout'] = entropy_dropout
    lm_labels_results['top_k_entropy_rand_dir'] = entropy_rand_dir
    return lm_labels_results


def eval_task_performance(x_train, y_train, x_dev, y_dev, x_train_no_label, x_dev_no_label, task_type):
    task_results = {}
    if task_type in classification_tasks:
        acc = learn_cls(x_train, y_train, x_dev, y_dev)
        acc_inlp = learn_cls(x_train_no_label, y_train, x_dev_no_label, y_dev)
    else:
        acc = learn_reg_cls(x_train, y_train, x_dev, y_dev)
        acc_inlp = learn_reg_cls(x_train_no_label, y_train, x_dev_no_label, y_dev)

    task_results['task_acc_vanialla'] = acc
    task_results['task_acc_p'] = acc_inlp
    return task_results


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
        name=task_type + '_eval',
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

    if use_wandb:
        wandb.run.summary['n_classes'] = len(pos2ind)
        wandb.run.summary['majority'] = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))
        wandb.run.summary['removed_directions'] = n_coords

    proj_file = deprobe_dir + '/P.npy'

    if os.path.isfile(proj_file):
        P = np.load(proj_file)
    else:
        raise FileNotFoundError('projection file does not exists...')

    print('evaluating performance')

    device = arguments['--device']
    lm_results = eval_lm_performance(tokenizer, out_embed, bias, x_dev, words_dev, P,
                                     n_coords=n_coords, device=device)
    if task in classification_tasks:
        probable_labels = most_probable_label(words_train, y_train)
        prediction_variety_results = eval_topk_performance(tokenizer, out_embed, bias, x_dev, words_dev, P, probable_labels,
                                                           n_coords, y_train, k=20, device=device)
    else:
        prediction_variety_results = {}

    print('removing property from inputs')
    x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=min(len(y_train), int(arguments['--n'])))
    x_train_no_label = data_projection(x_train, P)
    x_dev_no_label = data_projection(x_dev, P)

    task_results = eval_task_performance(x_train, y_train, x_dev, y_dev, x_train_no_label, x_dev_no_label, task)

    all_results = {**lm_results, **prediction_variety_results, **task_results}
    if use_wandb:
        for k, val in all_results.items():
            wandb.run.summary[k] = val

    table_data = []
    ind = 0
    for i in range(int(arguments['--display_examples'])):
        for w, orig_y, P_y, y_label in sentence_prediction_example(sentences_dev[i], vecs_dev[i], labels_dev[i], P):
            table_data.append([w, orig_y, P_y, y_label, ind])
            ind += 1
        table_data.append(['-', '-', '-', '-', ind])
        ind += 1

    wandb.log({"examples": wandb.Table(data=table_data, columns=["word", "lm_word", "-p_word", "label", "index"])})
    df = pd.DataFrame(table_data, columns=["word", "lm_word", "-p_word", "label", "index"])
    df.to_csv(deprobe_dir + '/examples.tsv', sep='\t', index=False)
