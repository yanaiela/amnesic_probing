import pickle
from collections import defaultdict, Counter

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import SGDClassifier, Ridge
from torch.nn.functional import kl_div
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from amnesic_probing.debias.debias import get_debiasing_projection, debias_by_specific_directions, get_pls_projection
from amnesic_probing.debiased_finetuning.utils import define_network

classification_tasks = ['task', 'word_len', 'subword', 'vowel']


def get_lm_logits(x, w, b):
    logits = np.dot(w, x.T) + np.array([b]).repeat(x.shape[0], axis=0).T
    return logits


def get_lm_predictions(tokenizer, w, b, x):
    logits = get_lm_logits(x, w, b)
    y = logits.argmax(axis=0)
    return tokenizer.convert_ids_to_tokens(y)


def get_top_k_lm_predictions(tokenizer, w, b, x, k=20):
    logits = get_lm_logits(x, w, b)
    top_y = logits.argsort(axis=0)[-k:][::-1]
    top_words = []
    for top_k_per_word in top_y:
        top_k = tokenizer.convert_ids_to_tokens(top_k_per_word)
        top_words.append(top_k)
    return top_words


def get_top_k_lm_predictions_gpu(tokenizer, w, b, x, y, projection: np.ndarray = None, k=100, device: str = 'cpu'):
    network = define_network(w, b, projection_mat=projection, device=device)
    distribution = network.get_probs(x, y)[0]
    top_y = torch.tensor(distribution).to(device).topk(k=k, dim=1, largest=True, sorted=True).indices.cpu().numpy()
    top_words = []
    for top_k_per_word in top_y:
        top_k = tokenizer.convert_ids_to_tokens(top_k_per_word)
        top_words.append(top_k)
    return top_words


def get_lm_predictions_gpu(w, b, x, y, projection: np.ndarray = None, device: str = 'cpu'):
    network = define_network(w, b, projection_mat=projection, device=device)
    accuracy = network.eval(x, y)
    return accuracy


def get_lm_softmax_gpu(w, b, x, y, device: str):
    network = define_network(w, b, device=device)
    distribution = network.get_probs(x, y)
    return distribution


def data_projection(x, projection_matrix):
    return x.dot(projection_matrix)


def dropout_control(x, n_coord):
    all_indices = np.array(range(x.shape[1]))
    np.random.shuffle(all_indices)
    random_indices = all_indices[:n_coord]
    x_rand_dropout = x.copy()
    x_rand_dropout[:, random_indices] = 0
    return x_rand_dropout


def rand_direction_control(x, n_coord):
    dim = x.shape[1]
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(n_coord)]

    # finding the null-space of random directions
    rand_direction_p = debias_by_specific_directions(rand_directions, dim)

    # and projecting the original data into that space (to remove random directions)
    x_rand_direction = rand_direction_p.dot(x.T).T
    return x_rand_direction


def learn_cls(x_train, y_train, x_dev, y_dev):
    clf = SGDClassifier(warm_start=True, loss='log', n_jobs=-1, max_iter=10000, random_state=0, early_stopping=True)

    clf.fit(x_train, y_train)
    acc = clf.score(x_dev, y_dev)
    return acc


def learn_reg_cls(x_train, y_train, x_dev, y_dev):
    clf = Ridge(random_state=0)

    clf.fit(x_train, y_train)
    acc = clf.score(x_dev, y_dev)
    return acc


def learn_pls_cls(x_train, y_train, x_dev, y_dev):
    clf = PLSRegression(n_components=100)

    clf.fit(x_train, y_train)
    acc = clf.score(x_dev, y_dev)
    return acc


def read_files(vec_f, label_f, text_f=None, ignore_special_tokens=False):
    vecs = np.load(vec_f, allow_pickle=True)

    if ignore_special_tokens:
        vecs = np.array([x[1:-1] for x in vecs])

    with open(label_f, 'rb') as f:
        labels = pickle.load(f)

    if text_f:
        with open(text_f, 'rb') as f:
            sentences = pickle.load(f)
    else:
        sentences = None

    return vecs, labels, sentences


def get_projection_matrix(num_clfs, x_train, y_train, x_dev, y_dev,
                          majority_acc, max_iter=500, summary_writer=None):
    clf = SGDClassifier
    params = {'warm_start': True, 'loss': 'log', 'n_jobs': -1, 'max_iter': max_iter, 'random_state': 0,
              'early_stopping': True}
    dim = x_train.shape[1]

    P, _, _, all_projections, best_projection = get_debiasing_projection(clf, params, num_clfs, dim,
                                                                         is_autoregressive=True,
                                                                         min_accuracy=majority_acc,
                                                                         X_train=x_train, Y_train=y_train,
                                                                         X_dev=x_dev, Y_dev=y_dev,
                                                                         summary_writer=summary_writer)

    return P, all_projections, best_projection


def get_regression_projection_matrix(num_clfs, x_train, y_train, x_dev, y_dev, dim, majority_acc, summary_writer=None):
    clf = Ridge
    params = {'random_state': 0}

    projection_matrix, _, _, all_projections, best_projection = get_debiasing_projection(clf, params, num_clfs, dim,
                                                                                         is_autoregressive=True,
                                                                                         min_accuracy=0,
                                                                                         X_train=x_train, Y_train=y_train,
                                                                                         X_dev=x_dev, Y_dev=y_dev,
                                                                                         summary_writer=summary_writer)

    return projection_matrix, all_projections, best_projection


def get_regression_pls(num_clfs, x_train, y_train, x_dev, y_dev, dim, majority_acc, summary_writer=None):
    projection_matrix, all_projections, best_projection = get_pls_projection(num_clfs, x_train, y_train, x_dev, y_dev,
                                                                             summary_writer=summary_writer)
    return projection_matrix, all_projections, best_projection


def get_lm_vals(model_name):
    lm_model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    out_embed = lm_model.cls.predictions.decoder.weight.detach().cpu().numpy()
    bias = lm_model.cls.predictions.decoder.bias.detach().cpu().numpy()
    return lm_model, tokenizer, out_embed, bias


def predict_word(vec, projection, out_embed, bias, tokenizer):
    logits = np.dot(out_embed, vec) + bias
    am = tokenizer.convert_ids_to_tokens([logits.argmax()])[0]

    logits_P = np.dot(out_embed, projection.dot(vec.T).T) + bias
    amp = tokenizer.convert_ids_to_tokens([logits_P.argmax()])[0]

    return am, amp


def dkl(w, b, x_orig, x_diff):
    logits = get_lm_logits(x_orig, w, b)
    logits_diff = get_lm_logits(x_diff, w, b)

    probs = softmax(logits, axis=1)
    probs_diff = softmax(logits_diff, axis=1)
    dkl = entropy(probs, probs_diff, axis=1)
    dkl_mean = dkl.mean()
    return dkl_mean


def dkl_gpu(w, b, x_orig, x_diff, y, plain_probs: np.ndarray = None, device: str = 'cpu'):
    if plain_probs is None:
        probs = get_lm_softmax_gpu(w, b, x_orig, y, device=device)
    else:
        probs = plain_probs
    probs_diff = get_lm_softmax_gpu(w, b, x_diff, y, device=device)

    all_dkl = []
    for batch_prob, batch_prob_diff in tqdm(zip(probs, probs_diff)):
        batch_dkl = kl_div(torch.tensor(batch_prob_diff).float().to(device).log(),
                           torch.tensor(batch_prob).float().to(device), reduction='none')\
            .sum(axis=1).cpu().numpy()
        all_dkl.extend(batch_dkl)

    dkl_mean = np.mean(all_dkl)
    return dkl_mean, probs


def most_probable_label(words, labels):
    words_labels = defaultdict(list)

    for word, label in zip(words, labels):
        words_labels[word].append(label)

    most_probable_label_per_word = {}
    for word, label_list in words_labels.items():
        most_probable_label_per_word[word] = Counter(label_list).most_common(1)[0][0]
    return most_probable_label_per_word


def convert_words2labels(words, probable_labels, label2ind, most_common_label):

    labels_freqeuency = np.zeros(len(label2ind))
    for word in words:
        labels_freqeuency[label2ind[probable_labels.get(word, most_common_label)]] += 1
    return labels_freqeuency


def calc_entropy(x, y, y_labels, probable_labels, tokenizer, out_embed, bias, k, device):
    all_labels = list(set(y_labels))
    ind2label = dict(enumerate(all_labels))
    label2ind = {v: k for k, v in ind2label.items()}
    most_common_label = Counter(y_labels).most_common(1)[0][0]

    top_words = get_top_k_lm_predictions_gpu(tokenizer, out_embed, bias, x, y, None, k=k,
                                             device=device)
    all_dists = torch.tensor(
        [convert_words2labels(top_words[i], probable_labels, label2ind, most_common_label) for i in range(len(top_words))]).to(device)
    # this will be normlized to 2
    entropy_score = torch.distributions.Categorical(logits=all_dists).entropy().mean().cpu().numpy()
    return entropy_score
