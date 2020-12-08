import glob
import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

from amnesic_probing.encoders.bert_encoding import lm_encoding, bert_based_encoding, \
    lm_masked_encoding, tokenize_and_preserve_labels
from amnesic_probing.models.qa import BertForQuestionAnsweringDeprobe

models = {
    'bert-base-uncased': {
        'type': 'bert',
        'name': 'bert-base-uncased',
        'tokenizer': BertTokenizer,
        'model': BertForMaskedLM,
        'config': None,
        'lower': True,
        'mlm': True,
    },
    'roberta-base': {
        'type': 'bert',
        'name': 'roberta-base',
        'tokenizer': RobertaTokenizer,
        'model': RobertaForMaskedLM,
        'config': None,
        'lower': True,
        'mlm': True,
    },
    'qa': {
        'type': 'bert',
        'name': 'deepset/bert-base-cased-squad2',
        'tokenizer': BertTokenizer,
        'model': BertForQuestionAnsweringDeprobe,
        'config': BertConfig,
        'lower': False,
        'mlm': False,
    },
}


def get_pretrained_models(model_type):
    params = models[model_type]

    model_name = params['name']

    if params['config'] is not None:
        config = params['config'].from_pretrained(
            model_name,
        )
    else:
        config = None
    tokenizer = params['tokenizer'].from_pretrained(
        model_name,
        do_lower_case=params['lower'],
    )
    if params['mlm']:
        model = params['model'].from_pretrained(
            model_name,
            output_hidden_states=True,
            from_tf=False,
            config=config,
        )
    else:
        model = params['model'].from_pretrained(
            model_name,
            from_tf=False,
            config=config,
        )

    return model.eval(), tokenizer


def encode_text(data, encoder, tokenizer, masked=False, only_last_layer=True):
    encoded_vectors = defaultdict(list)
    encoded_labels = defaultdict(list)

    for i, datum in enumerate(tqdm(data)):
        tokens = datum['text']

        if type(encoder) in [BertForMaskedLM, RobertaForMaskedLM]:
            if masked:
                last_vecs, rep_vecs = lm_masked_encoding(' '.join(tokens), encoder, tokenizer,
                                                         only_last_layer=only_last_layer)

            else:
                last_vecs, rep_vecs = lm_encoding(' '.join(tokens), encoder, tokenizer, only_last_layer=only_last_layer)

            encoded_vectors['last_vec'].append(last_vecs)
            encoded_vectors['rep_vec'].append(rep_vecs)
        else:
            rep_vecs = bert_based_encoding(' '.join(tokens), encoder, tokenizer)
            encoded_vectors['rep_vec'].append(rep_vecs)

        # going over all labels that were collected from the dataset
        for label_name, labels in datum['labels'].items():
            tok_sen, tok_label = tokenize_and_preserve_labels(tokens, labels, tokenizer)
            encoded_labels[label_name].append(tok_label)

        # tok_sen are the tokens of the current sentence. It doesn't change over the previous loop
        # therefore the last one equals all the rest (it assumes that there's at least one label)
        encoded_labels['tokens'].append(tok_sen)

    return {'vectors': encoded_vectors, 'labels': encoded_labels}


def to_file(encoded_data, output_dir, only_last_layer):
    if not os.path.isdir(output_dir):
        print('creating dir ', output_dir)
        os.makedirs(output_dir)

    for name, vals in encoded_data['vectors'].items():

        if name == "rep_vec" and not only_last_layer:
            for layer in range(len(vals[0])):
                X = np.array([x[layer] for x in vals])
                np.save(output_dir + '/vec_layer:{}.npy'.format(layer), X)
        else:
            np.save(output_dir + '/{}.npy'.format(name), np.array(vals))

    for name, vals in encoded_data['labels'].items():
        with open(output_dir + '/{}.pickle'.format(name), 'wb') as f:
            pickle.dump(vals, f)


def read_conll_format(input_file):
    data = []
    with open(input_file, 'r') as f:
        sen = []
        tag = []
        pos = []
        dep = []
        orig_vals = []
        for line in tqdm(f):
            if line.strip() == '':
                pos_next_word = pos[1:] + ['EOL']
                tag_next_word = tag[1:] + ['EOL']
                data.append({'text': sen,
                             'labels': {
                                 'tag': tag,
                                 'pos': pos,
                                 'dep': dep,
                                 'orig_vals': orig_vals,

                                 'pos_next_word': pos_next_word,
                                 'tag_next_word': tag_next_word
                             }
                             })
                sen = []
                tag = []
                pos = []
                dep = []
                orig_vals = []
                continue
            vals = line.split('\t')
            sen.append(vals[1])
            tag.append(vals[3])
            pos.append(vals[4])
            dep.append(vals[7])
            orig_vals.append(vals)

    return data


def read_onto_notes_format(input_file):
    data = []
    for cur_file in tqdm(glob.glob(input_file + '/data/english/**/*.*gold_conll', recursive=True)):

        with open(cur_file, 'r') as in_f:
            sen = []
            ner = []
            np_start = []
            np_end = []
            phrase_start = []
            phrase_end = []
            prev_ner = ''
            for line in in_f:
                if line.startswith('#'):
                    continue
                if line.strip() == '':
                    data.append({'text': sen,
                                 'labels': {
                                     'ner': ner,
                                     'phrase_start': phrase_start,
                                     'phrase_end': phrase_end,
                                     'np_start': np_start,
                                     'np_end': np_end,
                                 }
                                 })
                    sen = []
                    ner = []
                    np_start = []
                    np_end = []
                    phrase_start = []
                    phrase_end = []
                    continue
                vals = line.split()
                sen.append(vals[3])

                cur_ner = vals[10]
                if cur_ner.startswith('('):
                    cur_ner = cur_ner[1:]
                    prev_ner = cur_ner
                if cur_ner.endswith(')'):
                    cur_ner = prev_ner[:-1]
                    prev_ner = ''
                if prev_ner != '':
                    cur_ner = prev_ner
                if cur_ner != '*' and cur_ner.endswith('*'):
                    cur_ner = cur_ner[:-1]
                ner.append(cur_ner)

                constituency = vals[5]

                if '(NP' in constituency:
                    np_start.append('S')
                else:
                    np_start.append('NS')

                if 'NP)' in constituency:
                    np_end.append('E')
                else:
                    np_end.append('NE')

                if constituency.startswith('('):
                    phrase_start.append('S')
                else:
                    phrase_start.append('NS')

                if constituency.endswith(')'):
                    phrase_end.append('E')
                else:
                    phrase_end.append('NE')

    return data


def read_sem_tagging_format(input_file):
    """
    https://www.aclweb.org/anthology/W17-6901/
    https://arxiv.org/abs/1609.07053
    """
    data = []
    with open(input_file, 'r') as f:
        sen = []
        semtag = []
        orig_vals = []
        for line in tqdm(f):
            if line.strip() == '':
                data.append({'text': sen,
                             'labels': {
                                 'semtag': semtag,
                                 'orig_vals': orig_vals,
                             }
                             })
                sen = []
                semtag = []
                orig_vals = []
                continue
            vals = line.split('\t')
            sen.append(vals[1])
            semtag.append(vals[0])
            orig_vals.append(vals)

    return data


def read_coarse_sem_tagging_format(input_file):
    """
    https://www.aclweb.org/anthology/W17-6901/
    https://arxiv.org/abs/1609.07053
    """
    mapping = COARSE_SEMTAG_MAPPING

    data = []
    with open(input_file, 'r') as f:
        sen = []
        semtag = []
        orig_vals = []
        for line in tqdm(f):
            if line.strip() == '':
                data.append({'text': sen,
                             'labels': {
                                 'semtag': semtag,
                                 'orig_vals': orig_vals,
                             }
                             })
                sen = []
                semtag = []
                orig_vals = []
                continue
            vals = line.split('\t')
            sen.append(vals[1])
            semtag.append(mapping[vals[0]])
            orig_vals.append(vals)

    return data


def read_fce_format(input_file):
    """
    Compositional Sequence Labeling Models for Error Detection in Learner Writing
    Marek Rei and Helen Yannakoudakis
    In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)

    A New Dataset and Method for Automatically Grading ESOL Texts
    Helen Yannakoudakis, Ted Briscoe and Ben Medlock
    In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL-2011)

    """
    data = []
    with open(input_file, 'r') as f:
        sen = []
        grammatical = []
        orig_vals = []
        for line in tqdm(f):
            if line.strip() == '':
                data.append({'text': sen,
                             'labels': {
                                 'grammatical': grammatical,
                                 'orig_vals': orig_vals,
                             }
                             })
                # print(sen)
                sen = []
                grammatical = []
                orig_vals = []
                continue
            vals = line.split('\t')
            grammatical.append(vals[1])
            sen.append(vals[0])
            orig_vals.append(vals)

    return data


def read_coord_format(input_file):
    """
    True if the token is part of any COORD sub-tree, false otherwise

    python amnesic_probing/encoders/encode.py --input_file data/CoordinationExtPTB/train.txt --output_dir=out/coord/train
     --encoder=bert-base-uncased --encode_format=normal --format=coord
    """
    from nltk.tree import Tree

    def parse_coord(t: Tree, is_coord: bool):
        if len(t.leaves()) == 1:
            if t.pos()[0][1] == '-NONE-':
                return []
            else:
                return [(t.leaves()[0], is_coord)]

        res = []
        for subtree in t:
            res += parse_coord(subtree, is_coord or "COORD" in subtree.label())

        return res

    data = []
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            parsed_sen = Tree.fromstring(line)

            parsed_coords = parse_coord(parsed_sen, False)

            data.append({'text': [v[0] for v in parsed_coords],
                         'labels': {
                             'coord': [v[1] for v in parsed_coords],
                             'orig_vals': line,
                         }
                         })
    return data


COARSE_SEMTAG_MAPPING = {}

# v0.7
COARSE_SEMTAG_MAPPING['PRO'] = 'ANA'
COARSE_SEMTAG_MAPPING['DEF'] = 'ANA'
COARSE_SEMTAG_MAPPING['HAS'] = 'ANA'
COARSE_SEMTAG_MAPPING['REF'] = 'ANA'
COARSE_SEMTAG_MAPPING['EMP'] = 'ANA'
COARSE_SEMTAG_MAPPING['GRE'] = 'ACT'
COARSE_SEMTAG_MAPPING['ITJ'] = 'ACT'
COARSE_SEMTAG_MAPPING['HES'] = 'ACT'
COARSE_SEMTAG_MAPPING['QUE'] = 'ACT'
COARSE_SEMTAG_MAPPING['QUC'] = 'ATT'
COARSE_SEMTAG_MAPPING['QUV'] = 'ATT'
COARSE_SEMTAG_MAPPING['COL'] = 'ATT'
COARSE_SEMTAG_MAPPING['IST'] = 'ATT'
COARSE_SEMTAG_MAPPING['SST'] = 'ATT'
COARSE_SEMTAG_MAPPING['PRI'] = 'ATT'
COARSE_SEMTAG_MAPPING['DEG'] = 'ATT'
COARSE_SEMTAG_MAPPING['INT'] = 'ATT'
COARSE_SEMTAG_MAPPING['REL'] = 'ATT'
COARSE_SEMTAG_MAPPING['SCO'] = 'ATT'
COARSE_SEMTAG_MAPPING['EQU'] = 'COM'
COARSE_SEMTAG_MAPPING['MOR'] = 'COM'
COARSE_SEMTAG_MAPPING['LES'] = 'COM'
COARSE_SEMTAG_MAPPING['TOP'] = 'COM'
COARSE_SEMTAG_MAPPING['BOT'] = 'COM'
COARSE_SEMTAG_MAPPING['ORD'] = 'COM'
COARSE_SEMTAG_MAPPING['CON'] = 'UNE'
COARSE_SEMTAG_MAPPING['ROL'] = 'UNE'
COARSE_SEMTAG_MAPPING['GRP'] = 'UNE'
COARSE_SEMTAG_MAPPING['DXP'] = 'DXS'
COARSE_SEMTAG_MAPPING['DXT'] = 'DXS'
COARSE_SEMTAG_MAPPING['DXD'] = 'DXS'
COARSE_SEMTAG_MAPPING['ALT'] = 'LOG'
COARSE_SEMTAG_MAPPING['XCL'] = 'LOG'
COARSE_SEMTAG_MAPPING['NIL'] = 'LOG'
COARSE_SEMTAG_MAPPING['DIS'] = 'LOG'
COARSE_SEMTAG_MAPPING['IMP'] = 'LOG'
COARSE_SEMTAG_MAPPING['AND'] = 'LOG'
COARSE_SEMTAG_MAPPING['NOT'] = 'MOD'
COARSE_SEMTAG_MAPPING['NEC'] = 'MOD'
COARSE_SEMTAG_MAPPING['POS'] = 'MOD'
COARSE_SEMTAG_MAPPING['SUB'] = 'DSC'
COARSE_SEMTAG_MAPPING['COO'] = 'DSC'
COARSE_SEMTAG_MAPPING['APP'] = 'DSC'
COARSE_SEMTAG_MAPPING['BUT'] = 'DSC'
COARSE_SEMTAG_MAPPING['PER'] = 'NAM'
COARSE_SEMTAG_MAPPING['GPE'] = 'NAM'
COARSE_SEMTAG_MAPPING['GPO'] = 'NAM'
COARSE_SEMTAG_MAPPING['GEO'] = 'NAM'
COARSE_SEMTAG_MAPPING['ORG'] = 'NAM'
COARSE_SEMTAG_MAPPING['ART'] = 'NAM'
COARSE_SEMTAG_MAPPING['HAP'] = 'NAM'
COARSE_SEMTAG_MAPPING['UOM'] = 'NAM'
COARSE_SEMTAG_MAPPING['CTC'] = 'NAM'
COARSE_SEMTAG_MAPPING['URL'] = 'NAM'
COARSE_SEMTAG_MAPPING['LIT'] = 'NAM'
COARSE_SEMTAG_MAPPING['NTH'] = 'NAM'
COARSE_SEMTAG_MAPPING['EXS'] = 'EVE'
COARSE_SEMTAG_MAPPING['ENS'] = 'EVE'
COARSE_SEMTAG_MAPPING['EPS'] = 'EVE'
COARSE_SEMTAG_MAPPING['EXG'] = 'EVE'
COARSE_SEMTAG_MAPPING['EXT'] = 'EVE'
COARSE_SEMTAG_MAPPING['NOW'] = 'TNS'
COARSE_SEMTAG_MAPPING['PST'] = 'TNS'
COARSE_SEMTAG_MAPPING['FUT'] = 'TNS'
COARSE_SEMTAG_MAPPING['PRG'] = 'TNS'
COARSE_SEMTAG_MAPPING['PFT'] = 'TNS'
COARSE_SEMTAG_MAPPING['DAT'] = 'TIM'
COARSE_SEMTAG_MAPPING['DOM'] = 'TIM'
COARSE_SEMTAG_MAPPING['YOC'] = 'TIM'
COARSE_SEMTAG_MAPPING['DOW'] = 'TIM'
COARSE_SEMTAG_MAPPING['MOY'] = 'TIM'
COARSE_SEMTAG_MAPPING['DEC'] = 'TIM'
COARSE_SEMTAG_MAPPING['CLO'] = 'TIM'

# update
COARSE_SEMTAG_MAPPING['PRO'] = 'ANA'
COARSE_SEMTAG_MAPPING['DEF'] = 'ANA'
COARSE_SEMTAG_MAPPING['HAS'] = 'ANA'
COARSE_SEMTAG_MAPPING['REF'] = 'ANA'
COARSE_SEMTAG_MAPPING['EMP'] = 'ANA'
COARSE_SEMTAG_MAPPING['GRE'] = 'ACT'
COARSE_SEMTAG_MAPPING['ITJ'] = 'ACT'
COARSE_SEMTAG_MAPPING['HES'] = 'ACT'
COARSE_SEMTAG_MAPPING['QUE'] = 'ACT'
COARSE_SEMTAG_MAPPING['QUA'] = 'ATT'
COARSE_SEMTAG_MAPPING['UOM'] = 'ATT'
COARSE_SEMTAG_MAPPING['IST'] = 'ATT'
COARSE_SEMTAG_MAPPING['REL'] = 'ATT'
COARSE_SEMTAG_MAPPING['RLI'] = 'ATT'
COARSE_SEMTAG_MAPPING['SST'] = 'ATT'
COARSE_SEMTAG_MAPPING['PRI'] = 'ATT'
COARSE_SEMTAG_MAPPING['INT'] = 'ATT'
COARSE_SEMTAG_MAPPING['SCO'] = 'ATT'
COARSE_SEMTAG_MAPPING['ALT'] = 'LOG'
COARSE_SEMTAG_MAPPING['EXC'] = 'LOG'
COARSE_SEMTAG_MAPPING['NIL'] = 'LOG'
COARSE_SEMTAG_MAPPING['DIS'] = 'LOG'
COARSE_SEMTAG_MAPPING['IMP'] = 'LOG'
COARSE_SEMTAG_MAPPING['AND'] = 'LOG'
COARSE_SEMTAG_MAPPING['BUT'] = 'LOG'
COARSE_SEMTAG_MAPPING['EQA'] = 'COM'
COARSE_SEMTAG_MAPPING['MOR'] = 'COM'
COARSE_SEMTAG_MAPPING['LES'] = 'COM'
COARSE_SEMTAG_MAPPING['TOP'] = 'COM'
COARSE_SEMTAG_MAPPING['BOT'] = 'COM'
COARSE_SEMTAG_MAPPING['ORD'] = 'COM'
COARSE_SEMTAG_MAPPING['PRX'] = 'DEM'
COARSE_SEMTAG_MAPPING['MED'] = 'DEM'
COARSE_SEMTAG_MAPPING['DST'] = 'DEM'
COARSE_SEMTAG_MAPPING['SUB'] = 'DIS'
COARSE_SEMTAG_MAPPING['COO'] = 'DIS'
COARSE_SEMTAG_MAPPING['APP'] = 'DIS'
COARSE_SEMTAG_MAPPING['NOT'] = 'MOD'
COARSE_SEMTAG_MAPPING['NEC'] = 'MOD'
COARSE_SEMTAG_MAPPING['POS'] = 'MOD'
COARSE_SEMTAG_MAPPING['CON'] = 'ENT'
COARSE_SEMTAG_MAPPING['ROL'] = 'ENT'
COARSE_SEMTAG_MAPPING['GPE'] = 'NAM'
COARSE_SEMTAG_MAPPING['PER'] = 'NAM'
COARSE_SEMTAG_MAPPING['LOC'] = 'NAM'
COARSE_SEMTAG_MAPPING['ORG'] = 'NAM'
COARSE_SEMTAG_MAPPING['ART'] = 'NAM'
COARSE_SEMTAG_MAPPING['NAT'] = 'NAM'
COARSE_SEMTAG_MAPPING['HAP'] = 'NAM'
COARSE_SEMTAG_MAPPING['URL'] = 'NAM'
COARSE_SEMTAG_MAPPING['EXS'] = 'EVE'
COARSE_SEMTAG_MAPPING['ENS'] = 'EVE'
COARSE_SEMTAG_MAPPING['EPS'] = 'EVE'
COARSE_SEMTAG_MAPPING['EFS'] = 'EVE'
COARSE_SEMTAG_MAPPING['EXG'] = 'EVE'
COARSE_SEMTAG_MAPPING['ENG'] = 'EVE'
COARSE_SEMTAG_MAPPING['EPG'] = 'EVE'
COARSE_SEMTAG_MAPPING['EFG'] = 'EVE'
COARSE_SEMTAG_MAPPING['EXT'] = 'EVE'
COARSE_SEMTAG_MAPPING['ENT'] = 'EVE'
COARSE_SEMTAG_MAPPING['EPT'] = 'EVE'
COARSE_SEMTAG_MAPPING['EFT'] = 'EVE'
COARSE_SEMTAG_MAPPING['ETG'] = 'EVE'
COARSE_SEMTAG_MAPPING['ETV'] = 'EVE'
COARSE_SEMTAG_MAPPING['EXV'] = 'EVE'
COARSE_SEMTAG_MAPPING['NOW'] = 'TNS'
COARSE_SEMTAG_MAPPING['PST'] = 'TNS'
COARSE_SEMTAG_MAPPING['FUT'] = 'TNS'
COARSE_SEMTAG_MAPPING['DOM'] = 'TIM'
COARSE_SEMTAG_MAPPING['YOC'] = 'TIM'
COARSE_SEMTAG_MAPPING['DOW'] = 'TIM'
COARSE_SEMTAG_MAPPING['MOY'] = 'TIM'
COARSE_SEMTAG_MAPPING['DEC'] = 'TIM'
COARSE_SEMTAG_MAPPING['CLO'] = 'TIM'
COARSE_SEMTAG_MAPPING['APX'] = 'APX'
