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
        'base_dir': 'data/ontonotes_output_projection_{}/{}/train/',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end',
                   'ner_control', 'np_start_control', 'phrase_start_control', 'phrase_end_control'],
        'task': 'task',
    },
    'ud': {
        'base_dir': 'data/ud_output_projection_{}/{}/train/',
        'labels': ['tag',
                   'tag_control'],
        'task': 'task',
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
        task_type = vals['task']
        for masking in ['normal', 'masked']:
            for task in vals['labels']:

                # using the task labels in the regular case, or random labels from the same data
                # when the task is generated on the fly (e.g. word_len)
                data_label = task
                if task_type != 'task':
                    data_label = 'np_start'

                base_dir = vals['base_dir'].format(masking, task)
                labels = f'{base_dir}/{data_label}.pickle'
                text = f'{base_dir}/tokens.pickle'
                cartesian_product.append([base_dir, labels, text, task_type])

    parallelize(nodes, cartesian_product,
                'amnesic_probing/runs/evaluate/run_layer_wise_lm.sh',
                on_gpu=True, dry_run=dry_run)
