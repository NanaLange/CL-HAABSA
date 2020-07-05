#!/usr/bin/env python
# encoding: utf-8

import os, sys

sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter, curr_batch_index
import numpy as np

import matplotlib.pyplot as plt

tf.set_random_seed(1)


def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    print('I am lcr_rot_alt.')
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'l' + str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'r' + str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    return prob, att_l, att_r, att_t_l, att_t_r


def main(train_path, test_path, accuracyOnt, test_size, remaining_size, sort, num_buckets,
         l2=0.0001):  # learning_rate=0.07,
    # keep_prob=0.4,
    # momentum=0.9):
    # print_config()
    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, name='word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)
        momentum = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw = tf.placeholder(tf.int32, [None])

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)

        alpha_fw, alpha_bw = None, None
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target,
                                                                 tar_len, keep_prob1, keep_prob2, l2,
                                                                 'all')

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss,
                                                                                                        global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.keep_prob1,
            FLAGS.keep_prob2,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        # save_dir = 'temp_model/babysteps2buckets'
        # saver = saver_func(save_dir)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        def curr_get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, learning, moment,
                                bucket, is_shuffle=True):
            for index in curr_batch_index(bucket, batch_size, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                    learning_rate: learning,
                    momentum: moment,
                }
                yield feed_dict, len(index)

        def eval_get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, bucket,
                                is_shuffle=True):
            for index in curr_batch_index(bucket, batch_size, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_tl, max_tr = None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None

        train_time = 0
        max_time = 0

        total_iter = 0
        cost_func_test = []
        cost_func_train = []
        cost_func_eval = []
        acc_func_train = []
        acc_func_test = []
        acc_func_eval = []
        alltrainacc = []

        # split data in num_buckets
        buckets = np.array_split(sort, num_buckets)
        bucket_number = 1

        lowest_val = 100
        best_train = 0.
        best_test = 0.
        best_iteration = 0

        for bucket in buckets:  # for every bucket of data
            print("bucket number:{}".format(bucket_number))
            np.random.shuffle(bucket)
            tmp = int(round(0.8 * len(bucket)))
            traindata = bucket[:tmp]
            evaldata = bucket[tmp:]

            # update the hyperparameters for every bucket, depending on num_buckets
            if bucket_number == 1:
                lr = 0.01
                keep_prob = 0.7
                mom = 0.85
            if bucket_number == 2:
                lr = 0.01
                keep_prob = 0.7
                mom = 0.85
            if bucket_number == 3:
                lr = 0.02
                keep_prob = 0.6
                mom = 0.95
            if bucket_number == 4:
                lr = 0.08
                keep_prob = 0.3
                mom = 0.9
            if bucket_number == 5:
                lr = 0.07
                keep_prob = 0.4
                mom = 0.99
            if bucket_number == 6:
                lr = 0.02
                keep_prob = 0.6
                mom = 0.9
            if bucket_number == 7:
                lr = 0.05
                keep_prob = 0.4
                mom = 0.9
            if bucket_number == 8:
                lr = 0.05
                keep_prob = 0.4
                mom = 0.95
            if bucket_number == 9:
                lr = 0.01
                keep_prob = 0.5
                mom = 0.99
            if bucket_number == 10: # all the instances
                lr = 0.01
                keep_prob = 0.6
                mom = 0.9

            print("Training instances: {}, validation instances: {}".format(len(traindata), len(evaldata)))

            i = 0
            converged = False
            all_evalloss = []
            all_evalacc = []
            while i < FLAGS.n_iter and converged == False:  # until convergence or until certain amount of iterations
                trainacc, trainloss, traincnt = 0., 0., 0
                start_time = time.time()
                for train, numtrain in curr_get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y,
                                                           tr_target_word,
                                                           tr_tar_len,
                                                           FLAGS.batch_size, keep_prob, keep_prob, lr, mom,
                                                           traindata):
                    # _, step = sess.run([optimizer, global_step], feed_dict=train)
                    _, _trainloss, step, summary, _trainacc = sess.run(
                        [optimizer, loss, global_step, train_summary_op, acc_num],
                        feed_dict=train)

                    train_summary_writer.add_summary(summary, step)
                    # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                    # sess.run(embed_update)
                    trainacc += _trainacc  # saver.save(sess, save_dir, global_step=step)
                    traincnt += numtrain
                    trainloss += _trainloss * numtrain

                elapsed_time = time.time() - start_time
                train_time += elapsed_time

                evalacc, evalcost, evalcnt = 0., 0., 0
                for eva, evalnum in eval_get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y,
                                                        tr_target_word, tr_tar_len, 2000, 1.0, 1.0, evaldata, False):
                    _evalloss, _evalacc = sess.run([loss, acc_num], feed_dict=eva)
                    evalacc += _evalacc
                    evalcost += _evalloss * evalnum
                    evalcnt += evalnum
                acc, cost, cnt = 0., 0., 0
                fw, bw, tl, tr, ty, py = [], [], [], [], [], []
                p = []
                for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                                te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                    if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                        _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                            [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob],
                            feed_dict=test)
                        fw += list(_fw)
                        bw += list(_bw)
                        tl += list(_tl)
                        tr += list(_tr)
                    else:
                        _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                            [loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r],
                            feed_dict=test)
                    ty = np.asarray(_ty)
                    py = np.asarray(_py)
                    p = np.asarray(_p)
                    fw = np.asarray(_fw)
                    bw = np.asarray(_bw)
                    tl = np.asarray(_tl)
                    tr = np.asarray(_tr)
                    acc += _acc
                    cost += _loss * num
                    cnt += num

                comacc, comcnt = 0., 0
                for com, comnum in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y,
                                                  tr_target_word, tr_tar_len, FLAGS.batch_size, 1.0, 1.0, False):
                    _comloss, _comacc, _cty, _cpy, _cp, _cfw, _cbw, _ctl, _ctr = sess.run(
                        [loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=com)
                    comacc += _comacc
                    comcnt += comnum

                print(
                    'all samples={}, correct prediction={}, training time={}, training time so far={}'.format(cnt, acc,
                                                                                                              elapsed_time,
                                                                                                              train_time))
                trainacc = trainacc / traincnt
                acc = acc / cnt
                evalacc = evalacc / evalcnt
                comacc = comacc / comcnt
                alltrainacc.append(comacc)
                totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
                cost = cost / cnt
                trainloss = trainloss / traincnt
                evalcost = evalcost / evalcnt
                cost_func_test.append(cost)
                cost_func_train.append(trainloss)
                cost_func_eval.append(evalcost)
                acc_func_test.append(acc)
                acc_func_train.append(trainacc)
                acc_func_eval.append(evalacc)
                print('Iter {}: mini-batch loss validation set={:.6f}, train loss={:.6f}, train acc={:.6f}, '
                      'validation acc={:6f} test acc={:.6f}, total train acc={:6f}'.format(i, evalcost, trainloss,
                                                                                           trainacc, evalacc, acc,
                                                                                           comacc))
                summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
                test_summary_writer.add_summary(summary, step)

                all_evalloss.append(evalcost)
                all_evalacc.append(evalacc)
                if i > 1:  # want to compare current validation accuracy with val acc previous iterations
                    if (all_evalacc[i] - all_evalacc[i - 1] < 0.001) and (
                            all_evalacc[i - 1] - all_evalacc[i - 2] < 0.001) \
                            and (all_evalacc[i - 2] - all_evalacc[i - 3] < 0.001):
                        converged = True
                # if (all_evalloss[i] - all_evalloss[i-1] > 0.00001) and (all_evalloss[i-1] - all_evalloss[i-2] > 0.00001) \
                #    and (all_evalloss[i-2] - all_evalloss[i-3] > 0.00001):
                #   converged = True

                if bucket_number == num_buckets:
                    if evalcost < lowest_val:
                        lowest_val = evalcost
                        best_test = acc
                        best_train = comacc
                        best_iteration = i

                i += 1
                total_iter += 1

            bucket_number += 1

        # Plotting chart of training and testing losses as a function of iterations
        iterations = list(range(total_iter))
        plt.plot(iterations, cost_func_train, label='Cost func train')
        plt.plot(iterations, cost_func_test, label='Cost func test')
        plt.plot(iterations, cost_func_eval, label='Cost func validation')
        plt.title('Model loss k={}'.format(num_buckets))
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend(['train', 'test', 'eval'], loc='upper left')
        plt.show()

        # Plotting chart of training and testing accuracies as a function of iterations
        iterations = list(range(total_iter))
        plt.plot(iterations, acc_func_train, label='Acc func train')
        plt.plot(iterations, acc_func_test, label='Cost func test')
        plt.plot(iterations, acc_func_eval, label='Acc func validation')
        plt.title('Model accuracy k={}'.format(num_buckets))
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.legend(['train', 'test', 'eval'], loc='upper left')
        plt.show()

        print(
            'Optimization Finished! Iteration:{}: Validation loss={}, validation accuracy={}, test accuracy={}, in-sample acc={}'.format(
                total_iter, evalcost, evalacc, acc, comacc))

        print("Lowest validation loss:{}, at iteration:{}, with out-of-sample acc:{} and in-sample acc:{}"
              .format(lowest_val, best_iteration, best_test, best_train))

        print('iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            total_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            l2
        ))

        print(acc_func_train)
        print(acc_func_test)
        print(acc_func_eval)
        print(cost_func_eval)
        print(alltrainacc)

        return acc


if __name__ == '__main__':
    tf.app.run()
