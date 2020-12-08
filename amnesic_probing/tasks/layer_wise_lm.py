"""
Usage:
  layer_wise_lm.py [--proj_vecs=PROJ_VECS] [--labels=LABELS] [--text=TEXT] [--task=TASK]
        [--n=N] [--device=DEVICE]
        [--wandb]

Options:
  -h --help                     show this help message and exit
  --proj_vecs=proj_vecs         input dir of all the projected layers
  --labels=LABELS               labels file. using the train path (and automatically also using the dev,
                                by replacing train by dev)
  --text=TEXT                   text file
  --task=TASK                   task type. between word_ind, sen_len, task [default: task]
  --n=N                         number of training examples [default: 100000]
  --device=DEVICE               cpu, cuda:0, cuda:1, ... [default: cpu]
  --wandb                       log using wandb

"""

import wandb
from docopt import docopt
from sklearn.utils import shuffle
from tqdm import tqdm

from amnesic_probing.tasks.data_preparation import get_appropriate_data
from amnesic_probing.tasks.utils import read_files, get_lm_predictions_gpu, learn_cls, learn_pls_cls, get_lm_vals, \
    classification_tasks


def layer_eval(vecs_train, vecs_dev, labels_train, labels_dev, y_ids, out_embed, bias, task, device):
    x_train = vecs_train
    x_dev = vecs_dev

    # probing acc on the last layer after projection from layer i (< last layer)
    if task in classification_tasks:
        task_acc = learn_cls(x_train, labels_train, x_dev, labels_dev)
    else:
        task_acc = learn_pls_cls(x_train, labels_train, x_dev, labels_dev)

    # lm prediction acc after projection from layer i (< last layer)
    base_acc = get_lm_predictions_gpu(out_embed, bias, vecs_dev, y_ids, device=device)

    return task_acc, base_acc


def eval_layers(in_vecs_dir, in_labels_f, in_texts_f, task, device, n):
    _, tokenizer, out_embed, bias = get_lm_vals('bert-base-uncased')

    tasks_results = []

    for from_layer in tqdm(range(13)):
        vecs_train_f = f'{in_vecs_dir}/from:{from_layer}.to:{13}.npy'
        labels_train_f = in_labels_f

        vecs_train, labels_train, sentences_train = read_files(vecs_train_f, labels_train_f, text_f=in_texts_f,
                                                               ignore_special_tokens=True)
        vecs_dev, labels_dev, sentences_dev = read_files(vecs_train_f.replace('train', 'dev'),
                                                         labels_train_f.replace('train', 'dev'),
                                                         text_f=in_texts_f.replace('train', 'dev'),
                                                         ignore_special_tokens=True)

        (x_train, y_train, words_train), (x_dev, y_dev, words_dev) = get_appropriate_data(task, vecs_train,
                                                                                          labels_train,
                                                                                          sentences_train,
                                                                                          vecs_dev, labels_dev,
                                                                                          sentences_dev)

        x_train, y_train = shuffle(x_train, y_train, random_state=0, n_samples=min(len(y_train), n))

        y_ids = tokenizer.convert_tokens_to_ids(words_dev)
        task_acc, lm_acc = layer_eval(x_train, x_dev, y_train, y_dev, y_ids, out_embed, bias, task, device)
        print(from_layer, task_acc, lm_acc)
        tasks_results.append([from_layer, task_acc, lm_acc])
    return tasks_results


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
        name=f'{task_type}_layer_wise_lm',
        project="amnesic_probing",
        tags=["layer_wise", "lm", task_type, classification_type],
        config=config,
    )


if __name__ == '__main__':
    arguments = docopt(__doc__)

    labels = arguments['--labels']
    n = int(arguments['--n'])
    proj_vecs = arguments['--proj_vecs']
    texts = arguments['--text']
    task = arguments['--task']

    use_wandb = arguments['--wandb']
    if use_wandb:
        log_wandb(arguments)

    results = eval_layers(proj_vecs, labels, texts, task, arguments['--device'], int(arguments['--n']))

    if use_wandb:
        wandb.log({'per_layer_deprobe': wandb.Table(data=results,
                                                    columns=['from', 'task_acc', 'lm_acc'])})
