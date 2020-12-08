from functools import lru_cache

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
    h = layer_representation
    states = []

    for layer in layers:
        h = layer(h)[0]
        states.append(h)

    last_state_after_batch_norm = model.cls.predictions.transform(states[-1]).detach().cpu().numpy()[0]

    for i, s in enumerate(states):
        states[i] = s.detach().cpu().numpy()[0]

    return np.array(states), last_state_after_batch_norm


def lm_encoding(text, lm_model, tokenizer, only_last_layer=True):
    device = next(lm_model.parameters()).device
    # Encode text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    with torch.no_grad():
        last_hidden_states = lm_model(input_ids)

    # [1] - the second tuple with all hidden layers,
    # [-1] - the last layer
    # [0] bs = 0

    if only_last_layer:
        rep_state = last_hidden_states[1][-1][0]  # [1: -1]
    else:
        rep_state = [last_hidden_states[1][i][0] for i in range(len(last_hidden_states[1]))]

    last_state_after_batch_norm = lm_model.cls.predictions.transform(rep_state if only_last_layer else rep_state[-1])

    # detach

    if only_last_layer:
        rep_state = rep_state.detach().cpu().numpy()
    else:
        rep_state = [x.detach().cpu().numpy() for x in rep_state]

    last_state_after_batch_norm = last_state_after_batch_norm.detach().cpu().numpy()

    # rep_state: a list of tensors. shape: (len, 768) if only_last_layer; else (num_layers, len, 768)
    # last_state_after_batch_norm: a tensor, shape: (len, 768)

    return last_state_after_batch_norm, rep_state
    # w/o the special tokens, of the first (currently only) sentence
    # return last_hidden_states[0, 1:-1, :].detach().numpy()


def lm_masked_encoding(text, lm_model, tokenizer: transformers.PreTrainedTokenizer, batch_size=8, only_last_layer=True):
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

    all_last_states_after_batch_norm = []
    all_rep_states = []

    with torch.no_grad():
        for batch in batches:
            tensors = batch['tensors']
            masked_token_ixs = batch['masked_token_ixs']
            last_hidden_states = lm_model(tensors)

            # [1] - the second tuple with all hidden layers,
            # [-1] - the last layer

            rep_state = last_hidden_states[1][-1] if only_last_layer else list(last_hidden_states[1])
            last_state_after_batch_norm = lm_model.cls.predictions.transform(
                rep_state if only_last_layer else rep_state[-1])  # SHAPE: (batch_size, len, emb)

            # to numpy

            last_state_after_batch_norm = last_state_after_batch_norm.detach().cpu().numpy()
            rep_state = rep_state.detach().cpu().numpy() if only_last_layer else np.array(
                [h.detach().cpu().numpy() for h in rep_state])
            if not only_last_layer:
                rep_state = np.swapaxes(rep_state, 0,
                                        1)  # now shape is (batch_size, num_layers, len, 768)
            last_state_after_batch_norm = list(last_state_after_batch_norm)

            assert len(last_state_after_batch_norm) == len(
                masked_token_ixs), f"{len(last_state_after_batch_norm)}:{len(masked_token_ixs)}"
            last_state_after_batch_norm = [emb[tok_ix] for emb, tok_ix in
                                           zip(last_state_after_batch_norm,
                                               masked_token_ixs)]  # for each element in the batch (emb), choose only the index of the masked token (given by tok_ix)
            # last_state_after_batch_norm is (batch_size, 768)

            rep_state = [emb[tok_ix] if only_last_layer else emb[:, tok_ix] for emb, tok_ix in
                         zip(rep_state, masked_token_ixs)]
            # shape: (batch_size, 768) if only_last_layer else (batch_size, num_layers, 768) 

            last_state_after_batch_norm = [np.expand_dims(x, 0) for x in last_state_after_batch_norm]
            rep_state = [np.expand_dims(x, 0) if only_last_layer else x for x in rep_state]

            all_last_states_after_batch_norm.extend(last_state_after_batch_norm)
            all_rep_states.extend(rep_state)

    all_last_states_after_batch_norm = np.concatenate(all_last_states_after_batch_norm, axis=0)
    if only_last_layer:
        all_rep_states = np.concatenate(all_rep_states, axis=0)
    else:
        all_rep_states = np.array(all_rep_states)
        all_rep_states = np.swapaxes(all_rep_states, 0, 1)

    # all_rep_states shape: (len, 768) if only_last_layer else (num_layers, len, 768)
    # all_last_states_after_batch_norm shape: (len, 768)

    return all_last_states_after_batch_norm, all_rep_states


def bert_based_encoding(text, encoder, tokenizer):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = encoder.encode(input_ids)

    # [1] - the second tuple with all hidden layers,
    # [-1] - the last layer
    # [0] bs = 0
    # [1: -1] ignoring the special characters that were added before and after the sentence (cls)
    rep_state = last_hidden_states  # [1: -1]

    return rep_state.detach().numpy()


@lru_cache(maxsize=None)
def word_tokenize(tokenizer, word):
    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)
    return tokenized_word, n_subwords


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    taken from: https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L118
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word, n_subwords = word_tokenize(tokenizer, word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
