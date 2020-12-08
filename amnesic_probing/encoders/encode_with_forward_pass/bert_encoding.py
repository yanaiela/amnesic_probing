from collections import defaultdict

import numpy as np
import torch
import transformers


def forward_from_specific_layer(model, layer_number: int, layer_representation: torch.Tensor):
    """
   :param model: a BertForMaskedLM model
   :param layer_representation: a torch tensor, dims: [1, seq length, 768]
   Return: 
           states, a numpy array. dims: [#LAYERS - layer_number, seq length, 768]
           last_state_after_batch_norm: np array, after batch norm. dims: [seq_length, 768]
   """

    layers = model.bert.encoder.layer[layer_number:]
    layers.append(model.cls.predictions.transform)

    h = layer_representation
    states = []

    for i, layer in enumerate(layers):
        
        h = layer(h)[0] if i != len(layers) -1 else layer(h)
        states.append(h)

    #states[-1] = states[-1].unsqueeze(0)
    
    for i, s in enumerate(states):
        states[i] = s.detach().cpu().numpy()

    states = np.array(states)
    
    for x in states:
        assert len(x.shape) == 3
    
    return states



def lm_encoding_with_projection(text, lm_model, tokenizer, layer2projs=None):

    device = next(lm_model.parameters()).device
    result_dict = defaultdict(dict)
    # Encode text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    
    with torch.no_grad():
        last_hidden_states = lm_model(input_ids)

    # [1] - the second tuple with all hidden layers,
    # [-1] - the last layer
    # [0] bs = 0

    rep_state = [last_hidden_states[1][i][0] for i in range(len(last_hidden_states[1]))]

    for layer in range(0, len(rep_state)):
        P = layer2projs[layer]
        states = rep_state[layer]# if layer != 0 else input_embds

        # removing properties on all tokens except special tokens (cls, sep)
        states_projected = torch.cat([states[:1], (states[1:-1] @ P), states[-1:]], dim=0)
        # states_projected = states @ P
        
        next_layers = forward_from_specific_layer(lm_model, layer, states_projected.unsqueeze(0)).squeeze(1)
        result_dict[layer]["next_layers"] = next_layers
        result_dict[layer]["layer_projected"] = states_projected.detach().cpu().numpy()

    return result_dict


def lm_masked_encoding_with_projection(text, lm_model, tokenizer: transformers.PreTrainedTokenizer, batch_size=8,
                                       layer2projs=None):
    """
    CUDA_VISIBLE_DEVICES=2 python amnesic_probing/encoders/encode.py
        --input_file=ud/en-universal-dev.conll
        --output_dir=out/conll_masked_dev
        --encoder=bert-base-uncased
        --format=conll
        --encode_format=masked
        --device=cuda:0
    """
    device = next(lm_model.parameters()).device
    result_dict = defaultdict(dict)
    # logger.warning(f"BERT device found to be {device}. "
    #                f"We assume the entire model is only on this device (no multiple gpus)!")

    input_ids = tokenizer.encode(text)

    masked_inputs = []

    for tok_ix in range(len(input_ids)):
        masked_input = list(input_ids)
        masked_input[tok_ix] = tokenizer.mask_token_id
        masked_inputs.append(torch.tensor(masked_input).unsqueeze(0).to(device))  # SHAPE: (1, len)

    # masked_inputs is (len, 1, len); masked_inputs[i][j] will be the representation of word j on the version where word i is maked.

    # starting from 1 [for i in range(1, ... ] because we're not interested in masking CLS
    batches = [{'tensors': torch.cat(masked_inputs[i:i + batch_size], dim=0),
                'masked_token_ixs': list(range(i, i + len(masked_inputs[i:i + batch_size])))}
               for i in range(0, len(input_ids), batch_size)]  # SHAPE: (batch_size, len)

    # batches is (num_batches_per_sent, batch_size, len)

    all_rep_states = defaultdict(list)

    with torch.no_grad():
        for batch in batches:
            tensors = batch['tensors']
            masked_token_ixs = batch['masked_token_ixs']

            last_hidden_states = lm_model(tensors)

            rep_state = list(last_hidden_states[1])  # (num_layers, batch, len, 768)

            # to numpy           
            # rep_state = np.array([h.detach().cpu().numpy() for h in rep_state])
            # rep_state = np.swapaxes(rep_state, 0, 1) # now shape is (batch_size, num_layers, len, 768)

            for layer in range(0, len(rep_state)):
                h = rep_state[layer]  # (batch, len, 768)
                P = layer2projs[layer]
                # h_projected = h @ P
                # removing properties on all tokens except special tokens (cls, sep)
                h_projected = torch.cat([h[:1], (h[1:-1] @ P), h[-1:]], dim=0)

                next_layers = forward_from_specific_layer(lm_model, layer,
                                                          h_projected)  # (num_remaining_layers, batch, len, 768)                   
                relevant_next_layers = np.array(
                    [[layer[i][mask_ind] for i, mask_ind in enumerate(masked_token_ixs)] for layer in next_layers])
                # shape is (remaining_layers, batch_size, 768). we took only the masked token from each element.

                relevant_next_layers = np.swapaxes(relevant_next_layers, 0, 1)
                all_rep_states[layer].append(relevant_next_layers)

    for layer in all_rep_states.keys():
        states_lst = all_rep_states[layer]
        layer_seq = np.concatenate(states_lst,
                                   axis=0)  # concatenate over the batch dim to reconstruct the full sequence
        layer_seq = np.swapaxes(layer_seq, 0, 1)
        result_dict[layer]["next_layers"] = layer_seq

    return result_dict


def bert_based_encoding(text, encoder, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = encoder.encode(input_ids)

    # [1] - the second tuple with all hidden layers,
    # [-1] - the last layer
    # [0] bs = 0
    # [1: -1] ignoring the special characters that were added before and after the sentence (cls)
    # import pdb;
    # pdb.set_trace()
    rep_state = last_hidden_states  # [1: -1]

    return rep_state.detach().numpy()
