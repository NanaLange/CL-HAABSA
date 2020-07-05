import nltk
from nltk.wsd import lesk
from sklearn.preprocessing import minmax_scale, LabelEncoder, OneHotEncoder
from nltk.corpus import sentiwordnet as sw
import numpy as np


def tokenize_data(data):
    lines = open(data).readlines()
    sentences = []
    tokenizer = nltk.RegexpTokenizer(r"\w+")

    for i in range(0, len(lines), 3):
        sentences.append(tokenizer.tokenize(lines[i]))
        sentences.append(tokenizer.tokenize(lines[i + 1]))
        sentences.append((lines[i + 2]).rstrip())

    return sentences


def get_sent_length(data):
    opinions = []
    targets = []
    sentiment = []
    op_length = []
    tar_length = []
    for i in range(0, len(data), 3):
        opinions.append(data[i])  # all the sentences (with T for target word)
        targets.append(data[i + 1])  # all the target words
        sentiment.append(data[i + 2])  # all the sentiments (in the form 1\n, -1\n or 0\n)
        op_length.append(len(data[i]))  # amount of words in the opinion
        tar_length.append(len(data[i + 1]))  # amount of words in the target
    cont_length = []  # amount of words in the context
    for k in range(0, len(op_length)):
        cont_length.append(op_length[k] - 1)

    for j in range(0, len(opinions)):  # gives the total amount of words in the sentences (opinion words + target words)
        if tar_length[j] > 1:
            op_length[j] += tar_length[j] - 1

    return opinions, targets, sentiment, np.asarray(op_length), np.asarray(tar_length), np.asarray(cont_length)


def complete_sentences(targets, opinions):  # gives the complete sentences (target words instead of 'T')
    temp = []
    for (t, s) in zip(targets, opinions):  # makes complete sentences
        tar_ind = s.index('T')  # finds the index of the target words (T) in the current sentence
        del s[tar_ind]  # delete the word T
        s.insert(tar_ind, t)  # insert the target words
        s = [*s[0:tar_ind], *s[tar_ind], *s[(tar_ind + 1):len(s)]]  # make it one list
        temp.append(s)
    return temp


def scaled_scores(score, length):
    scaled = []
    for s, l in zip(score, length):
        scaled.append(s / l)
    return np.asarray(scaled)


def get_categories(aspects):  # create one hot encoding for the different aspect categories
    aspect_cat = []
    with open(aspects, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            aspect_cat.append(lines[i])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(aspect_cat)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    one_hot_cat = np.array(onehot_encoded)
    return (one_hot_cat)


def main(data, aspects):
    sentences = tokenize_data(data)
    ops, tars, sent, sen_len, tar_len, cont_len = get_sent_length(sentences)

    complete_sent = complete_sentences(tars, ops)

    aspect_cat = get_categories(aspects)

    synsets = []  # lists of all synsets
    for s in complete_sent:
        tag = nltk.pos_tag(s)
        syns = []
        for w, t in tag:  # every word and associated POS tag in sentence s
            if t.startswith("J"):
                t = 'a'
                if lesk(s, w, t) is None:  # if with head cluster no synset, then check if with satellite synset
                    t = 's'
                syns.append(lesk(s, w, t))
            elif t.startswith("R"):
                t = 'r'
                syns.append(lesk(s, w, pos=t))
            elif t.startswith("N"):
                t = 'n'
                syns.append(lesk(s, w, pos=t))
            elif t.startswith("V"):
                t = 'v'
                syns.append(lesk(s, w, pos=t))
            else:
                syns.append(lesk(s, w))
        synsets.append(syns)

    pos = []
    neg = []
    obj = []

    for syn in synsets:  # check list of all synsets for every sentence
        pos_score, neg_score, obj_score = 0, 0, 0
        for t in syn:  # go through every
            if t is not None:
                scores = sw.senti_synset(t.name())
                pos_score += scores.pos_score()
                neg_score += scores.neg_score()
                obj_score += scores.obj_score()
        pos.append(pos_score)  # all positivity scores per sentence
        neg.append(neg_score)  # all negativity scores per sentence
        obj.append(obj_score)  # all objectivity scores per sentence

    abs_diff = []
    for p, n in zip(pos, neg):
        abs_diff.append(abs(p - n))

    pos = np.asarray(pos)
    neg = np.asarray(neg)
    obj = np.asarray(obj)
    abs_diff = np.asarray(abs_diff)

    # get scaled versions of the scores
    pos_scaled = scaled_scores(pos, sen_len)
    neg_scaled = scaled_scores(neg, sen_len)
    obj_scaled = scaled_scores(obj, sen_len)
    abs_scaled = scaled_scores(abs_diff, sen_len)
    tar_scaled = scaled_scores(tar_len, sen_len)
    cont_scaled = scaled_scores(cont_len, sen_len)

    # normalize the scores (between -1 and 1) and make all arrays vertical
    sen_len = sen_len.astype(float)  # need dtype float64 for scaling instead of int32
    sen_len = np.vstack(minmax_scale(sen_len))
    pos = np.vstack(minmax_scale(pos))
    neg = np.vstack(minmax_scale(neg))
    obj = np.vstack(minmax_scale(obj))
    abs_diff = np.vstack(minmax_scale(abs_diff))
    pos_scaled = np.vstack(minmax_scale(pos_scaled))
    neg_scaled = np.vstack(minmax_scale(neg_scaled))
    obj_scaled = np.vstack(minmax_scale(obj_scaled))
    abs_scaled = np.vstack(minmax_scale(abs_scaled))
    tar_len = tar_len.astype(float)
    tar_len = np.vstack(minmax_scale(tar_len))
    tar_scaled = np.vstack(minmax_scale(tar_scaled))
    cont_len = cont_len.astype(float)
    cont_len = np.vstack(minmax_scale(cont_len))
    cont_scaled = np.vstack(minmax_scale(cont_scaled))
    features = np.concatenate((sen_len, pos, neg, obj, abs_diff, pos_scaled, neg_scaled, obj_scaled, abs_scaled,
                               tar_len, tar_scaled, cont_len, cont_scaled, aspect_cat), axis=1)

    return features, sent
