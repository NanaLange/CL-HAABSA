# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
import pickle

import tensorflow as tf

import auxModel
import cabascModel
import lcrModel
import lcrModelAlt_hierarchical_v4_baby_steps
import lcrModelAlt_hierarchical_v4_hyper_opt
import lcrModelAlt_hierarchical_v4_one_pass
import lcrModelAlt_hierarchical_v4_trainevaltest
import lcrModelInverse
import lcrModelAlt
import sentiWordNet
import svmModel
import utils
from OntologyReasoner import OntReasoner
from loadData import *

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np
import sys

import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4


# main function
def main(_):
    loadData = False  # only for non-contextualised word embeddings.
    #   Use prepareBERT for BERT (and BERT_Large) and prepareELMo for ELMo
    useOntology = False  # When run together with runLCRROTALT, the two-step method is used
    runLCRROTALT = False

    runSVM = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    weightanalysis = False

    runLCRROTALT_v1 = False
    runLCRROTALT_v2 = False
    runLCRROTALT_v3 = False
    runLCRROTALT_v4 = True

    #curriculum_learning = True
    # if curriculum_learning = True, then choose either one_pass or baby_steps to be True as well!
    runOne_Pass = False
    runBaby_Steps = True
    if runOne_Pass or runBaby_Steps: # if baby steps or one pass, then automatically curriculum learning True as well to get the sorted indices.
        curriculum_learning = True

    # determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runLCRROTALT_v1 or runLCRROTALT_v2 or runLCRROTALT_v3 or runLCRROTALT_v4:
        backup = True
    else:
        backup = False

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87
    tf.reset_default_graph()

    if useOntology == True:
        print('Starting Ontology Reasoner')
        # in sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(backup, FLAGS.test_path_ont, runSVM)
        # out of sample accuracy
        # Ontology = OntReasoner()
        # accuracyInSampleOnt, remainingInSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
            print(test[0])
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    # Get curriculum learning scores, either the ones already saved, or new ones
    # Make sure that the instances in FLAGS.train_path_ont and FLAGS.train_path (and the two test sets) have the same order of their instances!
    if curriculum_learning == True:

        try:
            sort_ind = pickle.load(open(FLAGS.sorted_indices, "rb"))
        except:
            tr_features, tr_sent = sentiWordNet.main(FLAGS.train_path_ont, FLAGS.train_aspect_categories)
            te_features, te_sent = sentiWordNet.main(FLAGS.test_path_ont, FLAGS.test_aspect_categories)
            tr_sent = np.asarray(utils.change_y_to_onehot(tr_sent))
            te_sent = np.asarray(utils.change_y_to_onehot(te_sent))
            print(tr_features.shape)
            print(tr_sent.shape)
            print(te_features.shape)
            print(te_sent.shape)

            curr_scores = auxModel.main(tr_features, te_features, tr_sent, te_sent)
            tf.reset_default_graph()
            inds1 = np.arange(0, len(curr_scores))
            sort_ind = [x for _, x in sorted(zip(curr_scores, inds1))]
            pickle.dump(sort_ind, open(FLAGS.sorted_indices, "wb"))

    # LCR-Rot-hop model
    if runLCRROTALT == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v1 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v1.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v2 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v2.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v3 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v3.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v4 == True:
        if runOne_Pass:
            acc = lcrModelAlt_hierarchical_v4_one_pass.main(FLAGS.train_path, test,
                                                                                         accuracyOnt,
                                                                                         test_size,
                                                                                         remaining_size,
                                                                                         sort_ind, FLAGS.num_buckets)
            tf.reset_default_graph()
        elif runBaby_Steps == True:
            tf.reset_default_graph()
            acc = lcrModelAlt_hierarchical_v4_baby_steps.main(FLAGS.train_path, test, accuracyOnt,
                                                                                           test_size,
                                                                                           remaining_size,
                                                                                           sort_ind, FLAGS.num_buckets)
            tf.reset_default_graph()

        else:
            acc = lcrModelAlt_hierarchical_v4_trainevaltest.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, test, FLAGS.train_path,
                                                                            accuracyOnt,
                                                                            test_size,
                                                                            remaining_size)
            tf.reset_default_graph()


print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
