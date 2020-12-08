"""
Usage:
  run_eval_per_dim.py [--dry_run]

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
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end'],
    },
    'ud': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['dep', 'pos', 'tag', 'pos_next_word'],
    },
    'binary_tag': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['tag_verb', 'tag_noun', 'tag_adp', 'tag_det', 'tag_num', 'tag_punct',
                   'tag_prt', 'tag_conj', 'tag_adv', 'tag_pron', 'tag_adj', 'tag_other'],
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
        base_dir = vals['base_dir']
        vecs = base_dir + '/' + vals['vecs']
        text = base_dir + '/' + vals['text']
        for masking in ['normal', 'masked']:
            base_dir = vals['base_dir'].format(masking)
            vecs = base_dir + '/' + vals['vecs']
            text = base_dir + '/' + vals['text']
            for label in vals['labels']:
                data_label = label
                cartesian_product.append([vecs, f'{base_dir}/{data_label}.pickle', text,
                                          f'models/lm/{label}/{masking}/layer:last/'])

    parallelize(nodes, cartesian_product,
                'amnesic_probing/runs/evaluate/run_specific_eval.sh',
                on_gpu=True, dry_run=dry_run)
