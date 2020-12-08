"""
Usage:
  encode.py [--input_file=INPUT_FILE] [--output_dir=OUTPUT_DIR] [--encoder=ENCODER]
                    [--format=FORMAT] [--encode_format=ENCODE_FORMAT] [--device=DEVICE] [--all_layers]

Options:
  -h --help                     show this help message and exit
  --input_file=INPUT_FILE       input file. conll format
  --output_dir=OUTPUT_DIR       output directory where to write the output files
  --encoder=ENCODER             encoder. types: bert-base-uncased, qa, ...
  --format=FORMAT               data format: conll, ontonotes, semtagging, fce
  --encode_format=ENCODE_FORMAT     encoding: normal, masked
  --device=DEVICE               cpu, cuda:0, cuda:1, ... (default: cpu)
  --all_layers                  encode all layers
"""

from docopt import docopt

from amnesic_probing.encoders import get_pretrained_models, encode_text, to_file, read_conll_format, read_onto_notes_format, \
    read_sem_tagging_format, read_coarse_sem_tagging_format, read_fce_format, read_coord_format

if __name__ == '__main__':
    arguments = docopt(__doc__)
    only_last_layer = not arguments["--all_layers"]
    print("only last layer:", only_last_layer)
    
    encoder, tokenizer = get_pretrained_models(arguments['--encoder'])
    encoder = encoder.to(arguments['--device'])

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

    if arguments['--encode_format'] == 'normal':
        final_data = encode_text(data, encoder, tokenizer, masked=False, only_last_layer=only_last_layer)
    elif arguments['--encode_format'] == 'masked':
        final_data = encode_text(data, encoder, tokenizer, masked=True, only_last_layer=only_last_layer)
    else:
        raise Exception('Unsupported encoding type')

    to_file(final_data, arguments['--output_dir'], only_last_layer=only_last_layer)
