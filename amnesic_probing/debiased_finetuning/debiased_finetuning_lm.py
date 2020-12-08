"""
Usage:
  debiased_finetuning_lm.py [--train_path=TRAIN_PATH] [--encoder=ENCODER] [--input_dim=INPUT_DIM] [--out_dir=OUT_DIR]
  [--n_epochs=N_EPOCHS] [--debias=DEBIAS] [--device=DEVICE] [--wandb]

Options:
  -h --help                     show this help message and exit
  --train_path=TRAIN_PATH       input directory
  --encoder=ENCODER             encoder. types: bert-base-uncased, qa, ... [default: bert-base-uncased]
  --input_dim=INPUT_DIM         input dimension [default: 768]
  --out_dir=OUT_DIR             logs and outputs directory
  --n_epochs=N_EPOCHS           number of epochs to run the classifier [default: 20]
  --debias=DEBIAS               the debias projection matrix to use, or none for no debiasing [default: none]
  --device=DEVICE               cpu, cuda:0, cuda:1, ... [default: cpu]
  --wandb                       log using wandb

"""

import numpy as np
import wandb
from docopt import docopt

from amnesic_probing.debiased_finetuning.utils import define_network, load_data, flatten_tokens
from amnesic_probing.tasks.utils import get_lm_vals


def log_wandb(arguments):
    out_dir = arguments['--out_dir']
    if out_dir[-1] == '/':
        out_dir = out_dir[:-1]
    task_type_full = out_dir.split('models/lm/')[1]
    task_type = task_type_full.split('/')[0]
    masking = task_type_full.split('/')[1]
    dataset_full = arguments['--train_path'].split('data/')[1].split('/')[0]
    dataset = dataset_full.split('_output', 1)[0]
    debias = arguments['--debias']
    if debias == 'none':
        task_type = task_type + '_baseline'
    debias = task_type

    config = dict(
        property=task_type,
        encoder='bert-base-uncased',
        dataset=dataset,
        masking=masking,
        debias_property=debias
    )

    wandb.init(
        name=task_type + '_ft_reg',
        project="amnesic_probing",
        tags=["lm", "emb_ft", task_type],
        config=config,
    )


if __name__ == '__main__':
    arguments = docopt(__doc__)

    use_wandb = arguments['--wandb']
    if use_wandb:
        log_wandb(arguments)

    _, tokenizer, word_embeddings, bias = get_lm_vals(arguments['--encoder'])

    train_dir = arguments['--train_path']
    dev_dir = train_dir.replace('train', 'dev')
    out_dir = arguments['--out_dir']
    input_dim = int(arguments['--input_dim'])

    if arguments['--debias'] == 'none':
        debias = np.eye(input_dim)
    else:
        debias = np.load(arguments['--debias'])

    train_vecs, train_labels = load_data(train_dir)
    dev_vecs, dev_labels = load_data(dev_dir)

    x_train, y_train = flatten_tokens(train_vecs, train_labels, tokenizer)
    assert len(x_train) == len(y_train), f"{len(x_train)}, {len(y_train)}"

    x_test, y_test = flatten_tokens(dev_vecs, dev_labels, tokenizer)
    assert len(x_test) == len(y_test), f"{len(x_test)}, {len(y_test)}"

    net = define_network(word_embeddings, bias, debias, arguments['--device'])

    if arguments['--debias'] == 'none':
        print("Debiasing option is turned off.")
        dev_acc = net.eval(x_test, y_test)
        print("Dev accuracy without fine-tuning is: ", dev_acc)

    net.train(x_train, y_train, x_test, y_test,
              epochs=int(arguments['--n_epochs']),
              save_path=f"{out_dir}/debias_ft.pt",
              use_wandb=use_wandb)
