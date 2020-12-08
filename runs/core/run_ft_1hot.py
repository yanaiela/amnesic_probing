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
    'ontonotes': {
        'train_dir': 'data/ontonotes_output_{}/train/',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end']
    },
    'ud': {
        'train_dir': 'data/ud_output_{}/train/',
        'labels': ['dep', 'pos', 'tag', 'pos_next_word']
    },
    'binary_tag': {
        'train_dir': 'data/ud_output_{}/train',
        'labels': ['tag_verb', 'tag_noun', 'tag_adp', 'tag_det', 'tag_num', 'tag_punct',
                   'tag_prt', 'tag_conj', 'tag_adv', 'tag_pron', 'tag_adj', 'tag_other'],
    },
    'binary_dep': {
        'train_dir': 'data/ud_output_{}/train',
        'labels': ['dep_adpmod', 'dep_det', 'dep_compmod', 'dep_num', 'dep_adpobj',
                   'dep_p', 'dep_poss', 'dep_adp', 'dep_amod', 'dep_nsubj',
                   'dep_dep', 'dep_dobj', 'dep_cc', 'dep_conj', 'dep_advmod',
                   'dep_ROOT', 'dep_ccomp', 'dep_aux', 'dep_xcomp', 'dep_neg'],
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
            for masking in ['normal', 'masked']:
                train_dir = vals['train_dir'].format(masking)
                cartesian_product.append([train_dir,
                                          f'models/lm/{label}/{masking}/layer:last/P.npy',
                                          f'{label}',
                                          f'models/lm/{label}/{masking}/layer:last/',
                                          ])

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/core/run_ft_1hot.sh',
                on_gpu=True, dry_run=dry_run)
