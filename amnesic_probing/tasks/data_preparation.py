import numpy as np

percentiles_list = [50]


def create_labeled_data(vecs, labels_seq, pos2i=None):
    x = []
    y = []

    if not pos2i:
        # using `sorted` function to make this process deterministic
        pos2i = {p: i for i, p in enumerate(sorted(set([item for sublist in labels_seq for item in sublist])))}

    for label, vec in zip(labels_seq, vecs):
        for l, v in zip(label, vec):
            x.append(v)
            y.append(pos2i[l])

    return np.array(x), np.array(y), pos2i


def create_subword_data(vecs, sentences_seq):
    x = []
    y = []

    for sentence, vec in zip(sentences_seq, vecs):
        for w, v in zip(sentence, vec):
            x.append(v)
            if w.startswith('##'):
                label = 1
            else:
                label = 0

            y.append(label)

    return np.array(x), np.array(y)


def create_word_length_data(vecs, sentences_seq):
    x = []
    y = []

    for sentence, vec in zip(sentences_seq, vecs):
        for w, v in zip(sentence, vec):
            x.append(v)
            if len(w) < 4:
                label = 'short'
            else:
                label = 'long'

            y.append(label)

    return np.array(x), np.array(y)


def create_character_data(vecs, sentences_seq):
    x = []
    y = []

    for sentence, vec in zip(sentences_seq, vecs):
        for w, v in zip(sentence, vec):
            x.append(v)
            if len(w) == 1 or w[1] in 'aeiou':
                label = 'vowel'
            else:
                label = 'consonant'

            y.append(label)

    return np.array(x), np.array(y)


def create_word_index_data(vecs, normalize=True):
    x = []
    y = []

    for vec in vecs:
        word_index = 0
        for v in vec:
            x.append(v)
            if normalize:
                y.append(float(word_index) / vec.shape[0])
            else:
                y.append(word_index)
            word_index += 1

    return np.array(x), np.array(y)


def get_label_in_percentile(val, percentiles):
    for i in range(len(percentiles)):
        if val < percentiles[i]:
            return i
    return len(percentiles)


def create_word_index_data_discrete(vecs, percentiles_sen_len=None):
    x = []
    y = []

    if percentiles_sen_len is None:
        sen_lens = [x for xx in [list(range(x.shape[0])) for x in vecs] for x in xx]
        percentiles_sen_len = np.percentile(sen_lens, percentiles_list)

    for vec in vecs:
        for word_index, v in enumerate(vec):
            x.append(v)
            y.append(get_label_in_percentile(word_index, percentiles_sen_len))

    return np.array(x), np.array(y), percentiles_sen_len


def create_sentence_len_data(vecs, normalize=True, max_sen_len=None):
    x = []
    y = []

    if not max_sen_len:
        max_sen_len = max([x.shape[0] for x in vecs])

    for vec in vecs:
        for v in vec:
            x.append(v)
            if normalize:
                y.append(vec.shape[0] / float(max_sen_len))
            else:
                y.append(vec.shape[0])

    return np.array(x), np.array(y), max_sen_len


def create_sentence_len_data_discrete(vecs, sen_len_percentiles=None):
    x = []
    y = []

    if sen_len_percentiles is None:
        sen_lens = [x for xx in [[x.shape[0]] * x.shape[0] for x in vecs] for x in xx]
        sen_len_percentiles = np.percentile(sen_lens, percentiles_list)

    for vec in vecs:
        sen_y = get_label_in_percentile(vec.shape[0], sen_len_percentiles)
        for v in vec:
            x.append(v)
            y.append(sen_y)

    return np.array(x), np.array(y), sen_len_percentiles


def filter_sen_len(vecs, labels, sentences):
    sen_lens = [len(x) for x in sentences]
    percentiles = np.percentile(sen_lens, [10, 95])
    f_vecs, f_labels, f_sentences = [], [], []
    for vec, label, sentence in zip(vecs, labels, sentences):
        sen_len = len(sentence)
        if sen_len < percentiles[0] or sen_len > percentiles[-1]:
            continue
        f_vecs.append(vec)
        f_labels.append(label)
        f_sentences.append(sentence)
    return f_vecs, f_labels, f_sentences


def get_appropriate_data(task_type, vecs_train, labels_train, sentences_train, vecs_dev, labels_dev, sentences_dev):

    if task_type == 'word_ind':
        # x_train, y_train, word_inds_percentiles = create_word_index_data_discrete(vecs_train)
        x_train, y_train = create_word_index_data(vecs_train)
        # x_dev, y_dev, _ = create_word_index_data_discrete(vecs_dev, word_inds_percentiles)
        x_dev, y_dev = create_word_index_data(vecs_dev)
    elif task_type == 'sen_len':
        # note that in there's an override of the previous train/dev sentences set.
        # This will also returned a filtered version of the data,
        # therefore, it might not be comparable to the other results, as the data is different
        vecs_train_f, labels_train_f, sentences_train = filter_sen_len(vecs_train, labels_train, sentences_train)
        vecs_dev_f, labels_dev_f, sentences_dev = filter_sen_len(vecs_dev, labels_dev, sentences_dev)

        # x_train, y_train, sen_len_percentiles = create_sentence_len_data_discrete(vecs_train)
        x_train, y_train, max_sen_len_train = create_sentence_len_data(vecs_train_f)
        # x_dev, y_dev, _ = create_sentence_len_data_discrete(vecs_dev, sen_len_percentiles=sen_len_percentiles)
        x_dev, y_dev, _ = create_sentence_len_data(vecs_dev_f, max_sen_len=max_sen_len_train)
    elif task_type == 'task':
        x_train, y_train, label2ind = create_labeled_data(vecs_train, labels_train)
        x_dev, y_dev, _ = create_labeled_data(vecs_dev, labels_dev, label2ind)
    elif task_type == 'subword':
        x_train, y_train = create_subword_data(vecs_train, sentences_train)
        x_dev, y_dev = create_subword_data(vecs_dev, sentences_dev)
    elif task_type == 'word_len':
        x_train, y_train = create_word_length_data(vecs_train, sentences_train)
        x_dev, y_dev = create_word_length_data(vecs_dev, sentences_dev)
    elif task_type == 'vowel':
        x_train, y_train = create_character_data(vecs_train, sentences_train)
        x_dev, y_dev = create_character_data(vecs_dev, sentences_dev)
    else:
        print('task: {} is not supported'.format(task_type))
        raise ValueError('task not supported')

    words_train = [w for sen in sentences_train for w in sen]
    words_dev = [w for sen in sentences_dev for w in sen]
    return (x_train, y_train, words_train), (x_dev, y_dev, words_dev)
