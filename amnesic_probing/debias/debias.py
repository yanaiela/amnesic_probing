import warnings
from typing import Dict
from typing import List

import numpy as np
import scipy
import wandb
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from tqdm import tqdm

from amnesic_probing.debias import classifier


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directions
    (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, best_iter_diff=0.01, summary_writer=None) \
        -> (np.ndarray, list, list, list, tuple):
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param best_iter_diff: float, diff from majority, used to decide on best iteration
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection;
            Ws, the list of all calssifiers.
    """

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []
    all_projections = []
    best_projection = None
    iters_under_threshold = 0
    prev_acc = -99.
    iters_no_change = 0

    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
        acc = clf.train_network(X_train_cp, Y_train, X_dev_cp, Y_dev)
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if summary_writer is not None:
            summary_writer.add_scalar('dev_acc', acc, i)
            wandb.log({'dev_acc': acc}, step=i)

        if iters_under_threshold >= 3:
            print('3 iterations under the minimum accuracy.. stopping the process')
            break

        if acc <= min_accuracy and best_projection is not None:
            iters_under_threshold += 1
            continue

        if prev_acc == acc:
            iters_no_change += 1
        else:
            iters_no_change = 0

        if iters_no_change >= 3:
            print('3 iterations with no accuracy change.. topping the process')
            break
        prev_acc = acc

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W)  # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far
             (instead of doing X = P_iX, which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1,
              due to e.g inexact argmin calculation).
            """
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            all_projections.append(P)

            # project
            X_train_cp = X_train.dot(P)
            X_dev_cp = X_dev.dot(P)

            # the first iteration that gets closest performance (or less) to majority
            if (acc - min_accuracy) <= best_iter_diff and best_projection is None:
                print('projection saved timestamp: {}'.format(i))
                best_projection = (P, i + 1)

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN)
    is roughly as accurate as this)
    """

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    if best_projection is None:
        print('projection saved timestamp: {}'.format(num_classifiers))
        print('using all of the iterations as the final projection')
        best_projection = (P, num_classifiers)

    return P, rowspace_projections, Ws, all_projections, best_projection


def get_debiasing_projection_by_cls(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                                    is_autoregressive: bool,
                                    min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                                    Y_dev: np.ndarray, by_class=True, Y_train_main=None,
                                    Y_dev_main=None, dropout_rate=0, summary_writer=None) -> (
        np.ndarray, list, list, list, tuple):
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only
           from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :param dropout_rate: float, default: 0 (note: not recommended to be used with autoregressive=True)
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection;
             Ws, the list of all calssifiers.
    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn(
            "Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []
    all_projections = []
    best_projection = None
    removed_directions = 0

    all_labels = list(set(Y_train))

    pbar = tqdm(range(len(all_labels)), ncols=600)
    for i in pbar:
        pbar_inner = tqdm(range(60), ncols=600)
        for j in pbar_inner:
            clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
            dropout_scale = 1. / (1 - dropout_rate + 1e-6)
            dropout_mask = (np.random.rand(*X_train.shape) < (1 - dropout_rate)).astype(float) * dropout_scale

            cur_cls_inds_train = np.where(Y_train == i)
            y_train = np.zeros(len(Y_train))
            y_train[cur_cls_inds_train] = 1
            cur_cls_inds_dev = np.where(Y_dev == i)
            y_dev = np.zeros(len(Y_dev))
            y_dev[cur_cls_inds_dev] = 1

            if len(cur_cls_inds_train[0]) == 0 or len(cur_cls_inds_dev[0]) == 0:
                continue

            acc = clf.train_network((X_train_cp * dropout_mask), y_train,
                                    X_dev_cp, y_dev)
            acc = np.around(acc, decimals=3)
            maj = ((len(y_dev) - len(cur_cls_inds_dev[0])) / float(len(y_dev)))
            maj = np.around(maj, decimals=3)
            pbar_inner.set_description("iteration: {}, accuracy: {}, cls: {}, majority: {}"
                                       .format(i, acc, i, maj))
            if summary_writer is not None:
                summary_writer.add_scalar('dev_acc', acc, i)
                wandb.log({'dev_acc': acc}, step=i)

            # if acc < min_accuracy: continue
            if abs(maj - acc) <= 0.005:
                break

            removed_directions += 1

            W = clf.get_weights()
            Ws.append(W)
            P_rowspace_wi = get_rowspace_projection(W)  # projection to W's rowspace
            rowspace_projections.append(P_rowspace_wi)

            if is_autoregressive:
                """
                to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far
                (instaed of doing X = P_iX, which is problematic when w_i is not exactly orthogonal
                 to w_i-1,...,w1, due to e.g inexact argmin calculation).
                """
                # use the intersection-projection formula of Ben-Israel 2013
                # http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
                # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

                P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
                all_projections.append(P)
                # project

                X_train_cp = X_train.dot(P)
                X_dev_cp = X_dev.dot(P)

                if abs(acc - min_accuracy) <= 0.01 and best_projection is None:
                    print('projection saved timestamp: {}'.format(i))
                    best_projection = (P, i)

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN)
     is roughly as accurate as this)
    """
    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    print('number of dimensions removed: {}'.format(removed_directions))

    if best_projection is None:
        print('projection saved timestamp: {}'.format(num_classifiers))
        best_projection = (P, num_classifiers)

    return P, rowspace_projections, Ws, all_projections, best_projection


def get_pls_projection(num_classifiers: int,
                       X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                       Y_dev: np.ndarray, summary_writer=None) \
        -> (np.ndarray, list, tuple):
    """
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param best_iter_diff: float, diff from majority, used to decide on best iteration
    :return: P, the debiasing projection; the list of all calssifiers.
    """

    all_projections = []
    best_projection = None
    iters_under_threshold = 0
    prev_acc = -99.
    iters_no_change = 0

    print('training pls')
    pls = PLSRegression(n_components=num_classifiers)
    pls.fit(X_train, Y_train)

    x_dim = X_train.shape[1]
    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        weights = pls.x_weights_[:, :i + 1]
        P = np.eye(x_dim, x_dim) - get_rowspace_projection(weights.T)
        all_projections.append(P)

        x_train_p = X_train.dot(P)
        x_dev_p = X_dev.dot(P)

        clf = Ridge(random_state=0)

        clf.fit(x_train_p, Y_train)
        acc = clf.score(x_dev_p, Y_dev)

        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if summary_writer is not None:
            summary_writer.add_scalar('dev_acc', acc, i)
            wandb.log({'dev_acc': acc}, step=i)

        # the first iteration that gets closest performance (or less) to majority
        # if (acc - min_accuracy) <= best_iter_diff and best_projection is None:
        #     print('projection saved timestamp: {}'.format(i))
        #     best_projection = (P, i + 1)

    # if best_projection is None:
    print('projection saved timestamp: {}'.format(num_classifiers))
    print('using all of the iterations as the final projection')
    best_projection = (all_projections[-1], num_classifiers)

    return P, all_projections, best_projection
