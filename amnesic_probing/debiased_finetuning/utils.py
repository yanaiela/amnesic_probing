import numpy as np
import torch
from amnesic_probing.debias.classifier import PytorchClassifier
import pickle


def define_network(W: np.ndarray, b: np.ndarray, projection_mat: np.ndarray = None, device: str = 'cpu'):
    embedding_net = torch.nn.Linear(in_features=W.shape[1], out_features=W.shape[0])
    embedding_net.weight.data = torch.tensor(W)
    embedding_net.bias.data = torch.tensor(b)

    if projection_mat is not None:
        projection_net = torch.nn.Linear(in_features=projection_mat.shape[1],
                                         out_features=projection_mat.shape[0],
                                         bias=False)
        projection_net.weight.data = torch.tensor(projection_mat, dtype=torch.float)
        for p in projection_net.parameters():
            p.requires_grad = False
        word_prediction_net = torch.nn.Sequential(projection_net, embedding_net)

    else:
        word_prediction_net = torch.nn.Sequential(embedding_net)

    net = PytorchClassifier(word_prediction_net, device=device)
    return net


def load_data(path):
    vecs = np.load(f"{path}/last_vec.npy", allow_pickle=True)
    vecs = np.array([x[1:-1] for x in vecs])

    with open(f"{path}/tokens.pickle", 'rb') as f:
        labels = pickle.load(f)

    return vecs, labels


def load_labels(labels_file):
    with open(labels_file, 'rb') as f:
        rebias_labels = pickle.load(f)

    return rebias_labels


def flatten_list(input_list):
    return [x for x_list in input_list for x in x_list]


def flatten_label_list(input_list, labels_list):
    flat_list = flatten_list(input_list)
    return np.array([labels_list.index(y) for y in flat_list]).flatten()


def flatten_tokens(all_vectors, all_labels, lm_tokenizer):
    x = np.array(flatten_list(all_vectors))
    y = np.array(
        [label for sentence_y in all_labels for label in
         lm_tokenizer.convert_tokens_to_ids(sentence_y)]).flatten()
    return x, y
