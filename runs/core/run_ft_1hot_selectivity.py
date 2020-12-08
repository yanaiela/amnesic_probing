"""
Usage:
  run_ft_1hot.py [--dry_run]

Options:
  -h --help                     show this help message and exit
  --dry_run                     if used, not running the experiments, and just printing them

"""

from docopt import docopt
from runs.ts_run import parallelize

# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    'nlp02',
]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘

runs_dic = {
    'ud': {
        'train_dir': 'data/ud_output_{}/train/',
        'labels': ['dep', 'tag']
    },
}

if __name__ == '__main__':
    arguments = docopt(__doc__)

    if arguments['--dry_run']:
        dry_run = True
    else:
        dry_run = False

    cartesian_product = []
    for data_type, vals in runs_dic.items():
        for label in vals['labels']:
            for masking in ['masked']:
                for iter in range(20):
                    train_dir = vals['train_dir'].format(masking)
                    cartesian_product.append([train_dir,
                                              f'models/lm/{label}/{masking}/layer:last/P_{iter}.npy',
                                              f'{label}',
                                              f'models/lm/{label}/{masking}/layer:last/',
                                              ])

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/core/run_ft_1hot.sh',
                on_gpu=True, dry_run=dry_run)
