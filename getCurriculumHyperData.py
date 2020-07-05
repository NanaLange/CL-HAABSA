import pickle

import numpy as np

from config import FLAGS

np.random.seed(123)
sort_ind = pickle.load(open(FLAGS.sorted_indices, "rb"))
num_buckets = range(1, 11)
for num in num_buckets:
    # split data in num_buckets
    buckets = np.array_split(sort_ind, num)
    lines = open(FLAGS.train_path).readlines()
    i = 0
    print("bucket number:{}".format(num))

    for bucket in buckets:
        i += 1
        print("amount of instances in bucket:{}".format(len(bucket)))
        np.random.shuffle(bucket)
        tmp = int(round(0.8 * len(bucket)))  # calculate how many instances for 80%
        train = bucket[:tmp]
        evals = bucket[tmp:]
        print("amount of instances train:{}".format(len(train)))
        print("amount of instances eval:{}".format(len(evals)))
        traindata = open(
            "data/programGeneratedData/curriculumLearning/curriculumhypertraindata_" + str(FLAGS.year) + "_" + str(
                num) + "_" + str(i) + "cross_train_10.txt", 'w')
        for t in train:
            traindata.write(lines[3 * t])
            traindata.write(lines[3 * t + 1])
            traindata.write(lines[3 * t + 2])
        traindata.close()

        evaldata = open(
            "data/programGeneratedData/curriculumLearning/curriculumhyperevaldata_" + str(FLAGS.year) + "_" + str(
                num) + "_" + str(i) + "cross_train_10.txt", 'w')
        for e in evals:
            evaldata.write(lines[3 * e])
            evaldata.write(lines[3 * e + 1])
            evaldata.write(lines[3 * e + 2])
        evaldata.close()

