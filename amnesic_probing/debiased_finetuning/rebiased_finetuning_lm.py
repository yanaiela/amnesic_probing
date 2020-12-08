"""
Usage:
  rebiased_finetuning_lm.py [--train_path=TRAIN_PATH] [--encoder=ENCODER] [--debias=DEBIAS] [--rebias=REBIAS]
  [--n_epochs=N_EPOCHS] [--out_dir=OUT_DIR] [--device=DEVICE] [--wandb]

Options:
  -h --help                     show this help message and exit
  --train_path=TRAIN_PATH       input directory
  --encoder=ENCODER             encoder. types: bert-base-uncased, qa, ... [default: bert-base-uncased]
  --debias=DEBIAS               the debias projection matrix to use, or none for no debiasing
  --rebias=REBIAS               the labels to add back to the representation
  --n_epochs=N_EPOCHS           number of epochs to run the classifier [default: 20]
  --out_dir=OUT_DIR             logs and outputs directory
  --device=DEVICE               device to use: cpu, cuda:0, cuda:1, ... [defaults: cpu]
  --wandb                       log using wandb

"""

import numpy as np
import torch
from docopt import docopt
import wandb

from amnesic_probing.debias.classifier import PytorchClassifier
from amnesic_probing.debiased_finetuning.utils import load_data, load_labels, flatten_tokens, flatten_label_list
from amnesic_probing.tasks.utils import get_lm_vals


class RebiasClassifier(torch.nn.Module):
    def __init__(self, debias, num_bias_labels, classifier):
        super(RebiasClassifier, self).__init__()

        debias_net = torch.nn.Linear(in_features=debias.shape[1], out_features=debias.shape[0], bias=False)
        debias_net.weight.data = torch.tensor(debias, dtype=torch.float)
        for p in debias_net.parameters():
            p.requires_grad = False
        net = torch.nn.Linear(in_features=classifier.shape[1], out_features=classifier.shape[0])
        net.weight.data = torch.tensor(classifier)
        net.bias.data = torch.tensor(bias)

        rebias_embedding_size = 32
        encode_rebias = torch.nn.Embedding(num_bias_labels, rebias_embedding_size)
        rebias_net = torch.nn.Linear(in_features=classifier.shape[1]+rebias_embedding_size,
                                     out_features=classifier.shape[1])

        self.debias_net = debias_net
        self.encode_rebias = encode_rebias
        self.rebias_net = rebias_net
        self.classifier_net = net

    def forward(self, input: torch.Tensor):
        rebias_labels = input[:, -1].long()
        input = input[:, :-1].float()

        debiased_input = self.debias_net(input)
        rebias = self.encode_rebias(rebias_labels)
        rebiased_input = torch.cat([debiased_input, rebias], dim=1)
        rebiased_input = self.rebias_net(rebiased_input)

        return self.classifier_net(rebiased_input)


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
    if len(debias.rsplit('/', 1)) > 1:
        proj = debias.rsplit('/', 1).split('.')[0]
        if '_' in proj:
            num = proj.split('_')[1]
            task_type = task_type + f'_iter:{num}'
    debias = task_type

    config = dict(
        property=task_type,
        encoder='bert-base-uncased',
        dataset=dataset,
        masking=masking,
        debias_property=debias
    )

    wandb.init(
        name=task_type + '_ft_1hot',
        project="amnesic_probing",
        tags=["lm", "1hot_emb_ft", task_type],
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

    if arguments['--debias'] == 'none':
        debias = np.eye(768)
    else:
        debias = np.load(arguments['--debias'])

    rebias_labels_name = arguments['--rebias']

    train_vecs, train_words = load_data(train_dir)
    train_labels = load_labels(f'{train_dir}/{rebias_labels_name}.pickle')

    dev_vecs, dev_words = load_data(train_dir.replace('train', 'dev'))
    dev_labels = load_labels(f'{dev_dir}/{rebias_labels_name}.pickle')

    all_labels = list(set([y for sen_y in train_labels for y in sen_y]))

    x_train, y_words_train = flatten_tokens(train_vecs, train_words, tokenizer)
    y_labels_train = flatten_label_list(train_labels, all_labels)
    assert len(x_train) == len(y_words_train), f"{len(x_train)}, {len(y_words_train)}"
    assert len(x_train) == len(y_labels_train), f"{len(x_train)}, {len(y_labels_train)}"

    x_dev, y_words_dev = flatten_tokens(dev_vecs, dev_words, tokenizer)
    y_labels_dev = flatten_label_list(dev_labels, all_labels)
    assert len(x_dev) == len(y_words_dev), f"{len(x_dev)}, {len(y_words_dev)}"
    assert len(x_dev) == len(y_labels_dev), f"{len(x_dev)}, {len(y_labels_dev)}"

    # adding bias label to input
    x_train = np.concatenate([x_train, y_labels_train.reshape(-1, 1)], axis=-1)
    x_dev = np.concatenate([x_dev, y_labels_dev.reshape(-1, 1)], axis=-1)

    out_dir = arguments['--out_dir']
    net = RebiasClassifier(debias=debias, num_bias_labels=len(all_labels), classifier=word_embeddings)
    net = PytorchClassifier(net, device=arguments['--device'])
    net.train(x_train, y_words_train, x_dev, y_words_dev,
              epochs=int(arguments['--n_epochs']),
              save_path=f"{out_dir}/finetuned_with_rebias.pt",
              use_wandb=use_wandb)
