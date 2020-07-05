from dataReader2016 import read_data_2016
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random


def loadDataAndEmbeddings(config, loadData):
    FLAGS = config

    if loadData == True:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data = read_data_2016(FLAGS.train_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                    FLAGS.train_path)
        print('reading test data...')
        test_data = read_data_2016(FLAGS.test_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                   FLAGS.test_path)

        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        word_embed = {}
        count = 0.0
        with open(FLAGS.pretrain_file, 'r', encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1

        print('finished embedding context vectors...')

        # print data to txt file
        outF = open(FLAGS.embedding_path, "w")
        for i, word in enumerate(source_word2idx):
            outF.write(word)
            outF.write(" ")
            outF.write(' '.join(str(w) for w in wt[i]))
            outF.write("\n")
        outF.close()
        print((len(source_word2idx) - count) / len(source_word2idx) * 100)

        return train_data[0], test_data[0], train_data[4], test_data[
            4]  # train_size, test_size, train_polarity_vector, test_polarity_vector

    else:
        # get statistic properties from txt file
        train_size, train_polarity_vector = getStatsFromFile(FLAGS.train_path)
        test_size, test_polarity_vector = getStatsFromFile(FLAGS.test_path)

        return train_size, test_size, train_polarity_vector, test_polarity_vector


def loadAverageSentence(config, sentences, pre_trained_context):
    FLAGS = config
    wt = np.zeros((len(sentences), FLAGS.edim))
    for id, s in enumerate(sentences):
        for i in range(len(s)):
            wt[id] = wt[id] + pre_trained_context[s[i]]
        wt[id] = [x / len(s) for x in wt[id]]

    return wt


def getStatsFromFile(path):
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        for i in range(0, len(lines), 3):
            # polarity
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


def loadHyperData(config, loadData, percentage=0.8):
    FLAGS = config

    if loadData:  # Gives both the BERT and the GloVe embeddings for the same instances and the associated aspect categories
        """Splits a file in 2 given the `percentage` to go in the large file."""
        random.seed(12345)
        with open(FLAGS.train_path, 'r') as fin1, \
                open(FLAGS.hyper_train_path, 'w') as foutBig1, \
                open(FLAGS.hyper_eval_path, 'w') as foutSmall1:
            with open(FLAGS.train_path_ont, 'r') as fin2, \
                    open(FLAGS.hyper_train_path_ont, 'w') as foutBig2, \
                    open(FLAGS.hyper_eval_path_ont, 'w') as foutSmall2:
                with open(FLAGS.train_aspect_categories, 'r') as fin3, \
                        open(FLAGS.hyper_train_aspect_categories, 'w') as foutBig3, \
                        open(FLAGS.hyper_eval_aspect_categories, 'w') as foutSmall3:

                    lines1 = fin1.readlines()
                    lines2 = fin2.readlines()
                    lines3 = fin3.readlines()

                    chunked1 = [lines1[i:i + 3] for i in range(0, len(lines1), 3)]
                    chunked2 = [lines2[i:i + 3] for i in range(0, len(lines2), 3)]

                    tmp = list(zip(chunked1, chunked2, lines3))
                    random.shuffle(tmp)
                    numlines = int(len(tmp) * percentage)
                    chunked1, chunked2, lines3 = zip(*tmp)
                    for chunk in chunked1[:numlines]:
                        for line in chunk:
                            foutBig1.write(line)
                    for chunk in chunked1[numlines:]:
                        for line in chunk:
                            foutSmall1.write(line)
                    for chunk in chunked2[:numlines]:
                        for line in chunk:
                            foutBig2.write(line)
                    for chunk in chunked2[numlines:]:
                        for line in chunk:
                            foutSmall2.write(line)
                    for line in lines3[:numlines]:
                        foutBig3.write(line)
                    for line in lines3[numlines:]:
                        foutSmall3.write(line)

        # with open(FLAGS.train_svm_path, 'r') as fin, \
        # open(FLAGS.hyper_svm_train_path, 'w') as foutBig, \
        # open(FLAGS.hyper_svm_eval_path, 'w') as foutSmall:
        #    lines = fin.readlines()

        #   chunked = [lines[i:i+4] for i in range(0, len(lines), 4)]
        #   random.shuffle(chunked)
        #   numlines = int(len(chunked)*percentage)
        #   for chunk in chunked[:numlines]:
        #       for line in chunk:
        #           foutBig.write(line)
        #   for chunk in chunked[numlines:]:
        #       for line in chunk:
        #           foutSmall.write(line)

    # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(FLAGS.hyper_train_path)
    test_size, test_polarity_vector = getStatsFromFile(FLAGS.hyper_eval_path)
    train_size_ont, train_polarity_vector_ont = getStatsFromFile(FLAGS.hyper_train_path_ont)

    return train_size, test_size, train_polarity_vector, test_polarity_vector,  # train_size_ont, train_polarity_vector_ont


def loadCrossValidation(config, split_size, load=True):
    FLAGS = config
    if load:
        words, svmwords, sent = [], [], []

        with open(FLAGS.train_path, encoding='cp1252') as f, \
                open(FLAGS.train_svm_path, encoding='cp1252') as svm:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                words.append([lines[i], lines[i + 1], lines[i + 2]])
                sent.append(lines[i + 2].strip().split()[0])
            words = np.asarray(words)

            svmlines = svm.readlines()
            for i in range(0, len(svmlines), 4):
                svmwords.append([svmlines[i], svmlines[i + 1], svmlines[i + 2], svmlines[i + 3]])
            svmwords = np.asarray(svmwords)

            sent = np.asarray(sent)

            i = 0
            kf = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=12345)
            for train_idx, val_idx in kf.split(words, sent):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                svmwords_1 = svmwords[train_idx]
                svmwords_2 = svmwords[val_idx]
                with open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_' + str(
                        i) + 'cross_train_10.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(
                            i) + 'cross_train_10.txt', 'w') as val, \
                        open("data/programGeneratedData/crossValidation" + str(
                            FLAGS.year) + '/svm/cross_train_svm_' + str(i) + 'cross_train_10.txt', 'w') as svmtrain, \
                        open(
                            "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/svm/cross_val_svm_' + str(
                                i) + 'cross_train_10.txt', 'w') as svmval:
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                    for row in words_2:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                    for row in svmwords_1:
                        svmtrain.write(row[0])
                        svmtrain.write(row[1])
                        svmtrain.write(row[2])
                        svmtrain.write(row[3])
                    for row in svmwords_2:
                        svmval.write(row[0])
                        svmval.write(row[1])
                        svmval.write(row[2])
                        svmval.write(row[3])
                i += 1
        # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(
        "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_0.txt')
    test_size, test_polarity_vector = [], []
    for i in range(split_size):
        test_size_i, test_polarity_vector_i = getStatsFromFile(
            "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(i) + 'cross_train_10.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector
