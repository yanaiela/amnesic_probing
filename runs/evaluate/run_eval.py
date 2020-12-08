"""
Usage:
  run_eval.py [--dry_run]

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
        'task_type': 'task'
    },
    'ud': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['dep', 'pos', 'tag', 'pos_next_word'],
        'task_type': 'task'
    },
    'binary_tag': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['tag_verb', 'tag_noun', 'tag_adp', 'tag_det', 'tag_num', 'tag_punct',
                   'tag_prt', 'tag_conj', 'tag_adv', 'tag_pron', 'tag_adj', 'tag_other'],
        'task_type': 'task'
    },
    'binary_dep': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['dep_adpmod', 'dep_det', 'dep_compmod', 'dep_num', 'dep_adpobj',
                   'dep_p', 'dep_poss', 'dep_adp', 'dep_amod', 'dep_nsubj',
                   'dep_dep', 'dep_dobj', 'dep_cc', 'dep_conj', 'dep_advmod',
                   'dep_ROOT', 'dep_ccomp', 'dep_aux', 'dep_xcomp', 'dep_neg'],
        'task_type': 'task'
    },
    'binary_ner': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        'text': 'tokens.pickle',
        'labels': ['ner_*', 'ner_ORG', 'ner_CARDINAL', 'ner_GPE', 'ner_DATE', 'ner_PERSON'],
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

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/evaluate/run_eval.sh',
                on_gpu=True, dry_run=dry_run)
