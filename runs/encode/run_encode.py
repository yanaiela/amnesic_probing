"""
Usage:
  run_encode.py [--dry_run]

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
        'out_dir': 'data/ontonotes_output_{}/',
        'task_format': 'ontonotes',
        'encode_format': ['normal', 'masked'],
    },
    'ud': {
        'splits': ['train', 'dev', 'test'],
        'input_file': 'data/ud/en-universal-{}.conll',
        'out_dir': 'data/ud_output_{}',
        'task_format': 'conll',
        'encode_format': ['normal', 'masked'],
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
                out_dir = vals['out_dir'].format(encoding_format)

                cartesian_product.append([split,
                                          input_file,
                                          out_dir,
                                          task_format,
                                          encoding_format,
                                          ])

    parallelize(nodes, cartesian_product, 'amnesic_probing/runs/encode/run_encode.sh',
                on_gpu=True, dry_run=dry_run)
