import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from transformers import BertForMaskedLM

from amnesic_probing.encoders.bert_encoding import tokenize_and_preserve_labels
from amnesic_probing.encoders.encode_with_forward_pass.bert_encoding import lm_encoding_with_projection, bert_based_encoding, \
    lm_masked_encoding_with_projection


def encode_text(data, encoder, tokenizer, masked=False, layer2projs=None, output_dir=None, batch_size = 512):

    batch_counter = 0
    encoded_vectors = defaultdict(list)
    encoded_labels = defaultdict(list)

    for i, datum in enumerate(tqdm(data)):
        
        #if i > 70: break
        
        tokens = datum['text']

        if type(encoder) == BertForMaskedLM:
            if masked:
                layer2next_layers = lm_masked_encoding_with_projection(' '.join(tokens), encoder, tokenizer,
                                                                       layer2projs=layer2projs)

            else:
                layer2next_layers = lm_encoding_with_projection(' '.join(tokens), encoder, tokenizer,
                                                                layer2projs=layer2projs)

            for key in layer2next_layers.keys():
                encoded_vectors[key].append(layer2next_layers[key])

        else:
            layer2next_layers = bert_based_encoding(' '.join(tokens), encoder, tokenizer)
            for key in layer2next_layers.keys():
                encoded_vectors[key].append(layer2next_layers[key])

        # going over all labels that were collected from the dataset
        for label_name, labels in datum['labels'].items():
            tok_sen, tok_label = tokenize_and_preserve_labels(tokens, labels, tokenizer)
            encoded_labels[label_name].append(tok_label)

        encoded_labels['tokens'].append(tok_sen)
        
        if i % batch_size == 0 and i > 0:
        
            encoding = {'vectors': encoded_vectors, 'labels': encoded_labels}
            to_file(encoding, output_dir, batch_counter)
            batch_counter += 1
            encoded_vectors = defaultdict(list)
            encoded_labels = defaultdict(list)
    
    # reminder of last batch
    if i % batch_size != 0:
        encoding = {'vectors': encoded_vectors, 'labels': encoded_labels}
        to_file(encoding, output_dir, batch_counter)    
        
                        
    unite_batched_files(output_dir)                
  
def unite_batched_files(output_dir):

    filenames = defaultdict(list)
    
    for filename in os.listdir(output_dir):
        if "batch=" not in filename: continue
        prefix = ".".join(filename.split(".")[:-1])
        
        batch = filename.split(".")[-1].split("=")[-1]
        filenames[prefix].append((int(batch), filename))

    for prefix, fnames in filenames.items():
    
        fnames = sorted(fnames, key = lambda batch_and_fname: batch_and_fname[0])
        all_data = []
        for batch, fname in fnames:
        
            data = np.load(output_dir + '/' + fname, allow_pickle = True)
            all_data.append(data)
        
        all_data = np.concatenate(all_data, axis = 0)
        with open(output_dir + "/" + prefix, "wb") as f:
            pickle.dump(all_data, f)
        
        for batch, fname in fnames:
            os.remove(output_dir + "/" + fname)

def to_file(encoded_data, output_dir, batch_counter):
    if not os.path.isdir(output_dir):
        print('creating dir ', output_dir)
        os.makedirs(output_dir)

    for layer, vals in encoded_data['vectors'].items():

        for j in range(len(vals[0]["next_layers"])):  # foreach next layer (over all sentences)
            X = np.array([x["next_layers"][j] for x in vals])
            layer_number = layer + j + 1
            path = output_dir + "from:{}.to:{}.npy.batch={}".format(layer, layer_number, batch_counter)

            with open(path, "wb") as f:
                np.save(f, X)

    for name, vals in encoded_data['labels'].items():
        with open(output_dir + '/{}.pickle.batch={}'.format(name, batch_counter), 'wb') as f:
            pickle.dump(vals, f)
