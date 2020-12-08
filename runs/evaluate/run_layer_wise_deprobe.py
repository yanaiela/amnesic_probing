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
        'layers_dir': 'data/ontonotes_output_{}/train/',
        'labels': ['ner', 'np_start', 'phrase_start', 'phrase_end'],
        'task_type': 'task'
    },
    'ud': {
        'base_dir': 'data/ud_output_projection_{}/{}/train/',
        'layers_dir': 'data/ud_output_{}/train/',
        'labels': ['tag'],
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
        for masking in ['normal', 'masked']:
            layer_dir = vals['layers_dir'].format(masking)
            for task in vals['labels']:
                base_dir = vals['base_dir'].format(masking, task)
                text_file = f'{base_dir}/tokens.pickle'
                task_type = vals['task_type']

                # using the task labels in the regular case, or random labels from the same data
                # when the task is generated on the fly (e.g. word_len)
                data_label = task
                if task_type != 'task':
                    data_label = 'np_start'
                cartesian_product.append([layer_dir,
                                          base_dir,
                                          f'{base_dir}/{data_label}.pickle',
                                          text_file,
                                          task_type])

    parallelize(nodes, cartesian_product,
                'amnesic_probing/runs/evaluate/run_layer_wise_deprobe.sh',
                on_gpu=False, dry_run=dry_run)
