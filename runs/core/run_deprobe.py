"""
Usage:
  run_deprobe.py [--dry_run]

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
# │ running combination  │
# └──────────────────────┘

runs_dic = {
    'ontonotes': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end'],
        'task': ['task'],
    },
    'ontonotes_layers': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'vec_layer:{}.npy',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end'],
        'layers': list(range(13)),
        'task': ['task'],
    },
    'ud': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'labels': ['dep', 'pos', 'tag', 'pos_next_word'],
        'task': ['task'],
    },
    'ud_layers': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'vec_layer:{}.npy',
        'labels': ['tag'],
        'layers': list(range(13)),
        'task': ['task'],
    },
    'binary_tag': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'labels': ['tag_verb', 'tag_noun', 'tag_adp', 'tag_det', 'tag_num', 'tag_punct',
                   'tag_prt', 'tag_conj', 'tag_adv', 'tag_pron', 'tag_adj', 'tag_other'],
        'task': ['task'],
        'balance': 'true'
    },
    'binary_dep': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        # used labels that appear at least 10K times
        'labels': ['dep_adpmod', 'dep_det', 'dep_compmod', 'dep_num', 'dep_adpobj',
                   'dep_p', 'dep_poss', 'dep_adp', 'dep_amod', 'dep_nsubj',
                   'dep_dep', 'dep_dobj', 'dep_cc', 'dep_conj', 'dep_advmod',
                   'dep_ROOT', 'dep_ccomp', 'dep_aux', 'dep_xcomp', 'dep_neg'],
        'task': ['task'],
        'balance': 'true',
    },
    'binary_ner': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        # used labels that appear at least 10K times
        'labels': ['ner_*', 'ner_ORG', 'ner_CARDINAL', 'ner_GPE', 'ner_DATE', 'ner_PERSON'],
        'task': ['task'],
        'balance': 'true',
    },
    'ontonotes_control': {
        'base_dir': 'data/ontonotes_output_{}/train',
        'vecs': 'last_vec.npy',
        'labels': ['ner_control', 'phrase_start_control', 'phrase_end_control'],
        'task': ['task'],
    },
    'ud_control': {
        'base_dir': 'data/ud_output_{}/train',
        'vecs': 'last_vec.npy',
        'labels': ['dep_control', 'pos_control', 'tag_control'],
        'task': ['task'],
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
        balanced = vals.get('balance', 'false')
        for masking in ['normal', 'masked']:
            base_dir = vals['base_dir'].format(masking)
            vecs = base_dir + '/' + vals['vecs']
            for task in vals['task']:
                for label in vals['labels']:
                    output_dir = 'models/lm/{0}/{1}/layer:{2}/'
                    if task == 'task':
                        task_name = label
                    else:
                        task_name = task
                    # running over all the model layers
                    if 'layers' in vals:
                        for layer in vals['layers']:
                            cartesian_product.append([vecs.format(layer),
                                                      f'{base_dir}/{label}.pickle',
                                                      output_dir.format(task_name, masking, layer),
                                                      task,
                                                      balanced],
                                                     )
                    else:
                        cartesian_product.append([vecs,
                                                  f'{base_dir}/{label}.pickle',
                                                  output_dir.format(task_name, masking, 'last'),
                                                  task,
                                                  balanced])

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/core/run_deprobe.sh',
                on_gpu=False, dry_run=dry_run)
