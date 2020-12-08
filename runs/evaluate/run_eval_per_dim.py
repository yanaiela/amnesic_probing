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
        'labels': ['ner', 'phrase_start', 'phrase_end'],
        'task_type': 'task'
    },
    'ud': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['dep', 'pos', 'tag'],
        'task_type': 'task'
    },
    'ontonotes_control': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['ner_control', 'phrase_start_control', 'phrase_end_control'],
        'task_type': 'task'
    },
    'ud_control': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['dep_control', 'pos_control', 'tag_control'],
        'task_type': 'task'
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
        task_type = vals['task_type']
        for masking in ['normal', 'masked']:
            base_dir = vals['base_dir'].format(masking)
            vecs = base_dir + '/' + vals['vecs']
            text = base_dir + '/' + vals['text']
            task_type = vals['task_type']
            for label in vals['labels']:
                data_label = label
                if task_type != 'task':
                    data_label = 'np_start'
                cartesian_product.append([vecs, f'{base_dir}/{data_label}.pickle', text,
                                          f'models/lm/{label}/{masking}/layer:last/',
                                          task_type])

    parallelize(nodes, cartesian_product,
                'amnesic_probing/runs/evaluate/run_eval_per_dim.sh',
                on_gpu=True, dry_run=dry_run)
