"""
Usage:
  run_layer_encode.py [--dry_run]

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
        'splits': ['train', 'dev', 'test'],
        'input_file': 'data/ontonotes/{}',
        'out_dir': 'data/ontonotes_output_projection_{}/{}/{}/',
        'task_format': 'ontonotes',
        'encode_format': ['normal', 'masked'],
        'tasks': ['ner', 'np_start', 'phrase_start', 'phrase_end', 'word_len', 'vowel'],
    },
    'ud': {
        'splits': ['train', 'dev', 'test'],
        'input_file': 'data/ud/en-universal-{}.conll',
        'out_dir': 'data/ud_output_projection_{}/{}/{}/',
        'task_format': 'conll',
        'encode_format': ['normal', 'masked'],
        'tasks': ['tag']
    },
    'regression': {
        'splits': ['dev', 'test', 'train'],
        'input_file': 'data/ontonotes/{}',
        'out_dir': 'data/ontonotes_output_projection_{}/{}/{}/',
        'task_format': 'ontonotes',
        'encode_format': ['normal', 'masked'],
        'tasks': ['word_ind'],
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
        task_format = vals['task_format']
        for split in vals['splits']:
            input_file = vals['input_file']
            input_file = input_file.format(split)
            for encoding_format in vals['encode_format']:
                for task in vals['tasks']:
                    for control in ['false', 'true']:
                        if control == 'true':
                            out_dir = vals['out_dir'].format(encoding_format, task + '_control', split)
                        else:
                            out_dir = vals['out_dir'].format(encoding_format, task, split)

                        cartesian_product.append([input_file,
                                                  f'models/lm/{task}/{encoding_format}/',
                                                  out_dir,
                                                  task_format,
                                                  encoding_format,
                                                  control
                                                  ])

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/encode/run_layer_encode.sh',
                on_gpu=True, dry_run=dry_run)
