"""
Usage:
  encode.py [--input_file=INPUT_FILE] [--projections_dir=PROJECTIONS_DIR] [--output_dir=OUTPUT_DIR] [--encoder=ENCODER]
                    [--format=FORMAT] [--encode_format=ENCODE_FORMAT] [--device=DEVICE] [--num_layers=NUM_LAYERS]
                    [--control=CONTROL]
                    [--batch_size=BATCH_SIZE]

Options:
  -h --help                     show this help message and exit
  --input_file=INPUT_FILE       input file. conll format
  --projections_dir=PROJECTIONS_DIR         directory where projection matrices are stored.
  --output_dir=OUTPUT_DIR       output directory where to write the output files
  --encoder=ENCODER             encoder. types: bert-base-uncased, qa, ... [default: bert-base-uncased]
  --format=FORMAT               data format: conll, ontonotes, semtagging, fce
  --encode_format=ENCODE_FORMAT     encoding: normal, masked
  --device=DEVICE               cpu, cuda:0, cuda:1, ... [default: cpu]
  --num_layers=NUM_LAYERS       how many layers does the model have [default: 12]
  --control=CONTROL             instead of using the learned projection, creating random ones, with the same amount of
                                directions. values: true|false [default: false]
  --batch_size=BATCH_SIZE       batch_size [default: 1024]

"""

import numpy as np
import torch
import json
from docopt import docopt

from amnesic_probing.encoders import get_pretrained_models, read_conll_format, read_onto_notes_format, \
    read_sem_tagging_format, read_coarse_sem_tagging_format, read_fce_format, read_coord_format
from amnesic_probing.encoders.encode_with_forward_pass import encode_text, to_file
from amnesic_probing.debias.debias import debias_by_specific_directions


def create_rand_dir_projection(dim, n_coord):
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(n_coord)]

    # finding the null-space of random directions
    rand_direction_projection = debias_by_specific_directions(rand_directions, dim)
    return rand_direction_projection


def get_random_projection(dir_path, vecs_dim=768):
    with open(dir_path + '/meta.json', 'r') as f:
        meta = json.load(f)
    n_coord = int(meta['removed_directions'])
    projection = create_rand_dir_projection(vecs_dim, n_coord)
    return projection


def load_projection_matrices(dir_path, num_layers, control: bool, device="cpu"):
    layer2proj = {}
    if dir_path is None:
        print("*** WARNING: PROJECTION DIR NOT SUPPLIED, USES RANDOM PROJECTION MATRICES ***")
        for layer in range(0, num_layers + 1):
            layer2proj[layer] = torch.eye(768).to(device).float() #torch.tensor(np.random.rand(768, 768)).to(device).float()

    else:
        for layer in range(0, num_layers + 1):
            if control:
                print('generating random projections')
                projection = get_random_projection(dir_path + f'/layer:{layer}/', vecs_dim=768)
            else:
                projection = np.load(dir_path + f'/layer:{layer}/P.npy')
            layer2proj[layer] = torch.tensor(projection).to(device).float()

    return layer2proj


if __name__ == '__main__':
    arguments = docopt(__doc__)

    #arguments = {"--encoder": "bert-base-uncased", "--device": "cuda", "--num_layers": 12, "--format": "conll", "--encode_format": "normal", "--input_file": "data/ud/en-universal-dev.conll", "--projections_dir": None, "--output_dir": "data/ud_output_projection_normal/train/", "--batch_size": 32}

    encoder, tokenizer = get_pretrained_models(arguments['--encoder'])
    encoder = encoder.to(arguments['--device'])
    print(arguments['--num_layers'])

    control = bool(arguments['--control'] == 'true')
    print('control projections: ', control)

    layer2projs = load_projection_matrices(arguments["--projections_dir"], int(arguments['--num_layers']), control,
                                           arguments['--device'])

    data_format = arguments['--format']
    if data_format == 'conll':
        data = read_conll_format(arguments['--input_file'])
    elif data_format == 'ontonotes':
        data = read_onto_notes_format(arguments['--input_file'])
    elif data_format == 'semtagging':
        data = read_sem_tagging_format(arguments['--input_file'])
    elif data_format == 'coarse_semtagging':
        data = read_coarse_sem_tagging_format(arguments['--input_file'])
    elif data_format == 'fce':
        data = read_fce_format(arguments['--input_file'])
    elif data_format == 'coord':
        data = read_coord_format(arguments['--input_file'])
    else:
        raise Exception('Unsupported file format exception')

    masked_encoding = arguments['--encode_format'] == 'masked'
    final_data = encode_text(data, encoder, tokenizer, masked=masked_encoding, layer2projs=layer2projs, output_dir = arguments['--output_dir'], batch_size = int(arguments["--batch_size"]))

    #to_file(final_data, )
