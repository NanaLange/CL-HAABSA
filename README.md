# CL-HAABSA
Implementation of curriculum learning (CL) in HAABSA++

Code for implementing curriculum learning in the two-step Hybrid Approach for Aspect-Based Sentiment Analysis (HAABSA) with contextual word embeddings and hierarchical attention (HAABSA++).
The HAABSA++ paper can be found via: https://arxiv.org/pdf/2004.08673.pdf.
The HAABSA paper can be found via: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf.

The HAABSA++ model uses a domain sentiment ontology and a neural network as backup. Curriculum learning (CL) is used to improve the results of the HAABSA++ model.


## Software Installation
First, the right environment needs to be set up and the right files need to be downloaded. This can be done by following the installation instructions given at https://github.com/ofwallaart/HAABSA. Only the files mentioned in the Read Me have to be downloaded, there is no need to download the HAABSA files. Hereafter, the right word embeddings need to be downloaded via: https://github.com/mtrusca/HAABSA_PLUS_PLUS. Again, only the files mentioned in the Read Me should be downloaded. 

After completing the instructions, all the CL-HAABSA files need to be installed into the newly created environment. An explanation of the CL-HAABSA files is given below.


## Software Explanation
The are three main files in the environment that can be run: main.py, main_cross.py, and main_hyper.py. An overview of these files and other important files is given:

- main.py: program to run in-sample and out-of-sample validation runs of the different methods. Each method can be activated by setting its corresponding boolean to True. For instance, set runBaby_Steps = True to run the *baby steps* curriculum algorithm.

- main_cross.py: program similar to main.py but for running a k-fold cross validation procedure for the chosen method.

- main_hyper.py: program to perform hyperparameter optimzation for a given space of hyperparameters for each method. The method can be changed by changing the objective and space parameters in the run_a_trial() function.

- config.py: file that contains parameter configurations that can be changed, such as the number of buckets used for curriculum strategy and the year of the data set.

- lcrModel.py, lcrModelAlt.py, lcrModelInverse.py: Tensorflow implementation for the LCR-Rot algorithm, the LCR-Rot-hop algorithm, and the LCR-Rot-inv algorithm, respectively. 

- lcrModelAlt_hierarchical_v1, lcrModelAlt_hierarchical_v2, lcrModelAlt_hierarchical_v3, and lcrModelAlt_hierarchical_v4: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention. The number stands for which method for hierarchical attention is used (see HAABSA++ paper). In CL-HAABSA++, we use an altered version of lcrModelAlt_hierarchical_v4, which is discussed next.

- lcrModelAlt_hierarchical_v4_trainevaltest.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4). Different than the methods discussed before, this program takes as input not only train and test data, but also validation data. Before running this method, the train data must be split in a train (80%) and a validation (20%) set. The validation data set is needed to decide when the model has converged

- lcrModelAlt_hierarchical_v4_baby_steps.py, lcrModelAlt_hierarchical_v4_one_pass.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4) with, respectively, the *baby steps* or the *one-pass* curriculum strategy. For every bucket of data, or combination of buckets, the optimal hyperparameters can be set in this file. Furthermore, the indices sorted by their curriculum scores need to be added as input. 

- lcrModelAlt_hierarchical_hyper_opt.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4) that can be used for the hyperparameter optimization of the buckets of training data. 

- getCurriculumHyperData.py: program that uses the curriculum scores to divide the training data in *k* buckets, for *k*=1,..,10. For every bucket of data, the data is separated in a train and an validation set (80-20%) and saved in new files. These files can then be used to optimize the hyperparameters for every bucket of data, using main_hyper.py.

- auxModel.py: the auxiliary feed-forward model used to compute the curriculum scores of the training data.

- hyperOpt.py: implementation of HyperOpt to optimize the hyperparameters of the auxiliary feed-forward model used for curriculum learning.

- sentiWordNet.py: file used to determine the SentiWordNet features for every sentence. These features can be fed into auxModel.py to obtain the curriculum scores for the instances. 

- att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.

- cabascModel.py: Tensorflow implementation for the CABASC algorithm

- svmModel.py: PYTHON implementation for a BoW model using a SVM.

- OntologyReasoner.py: PYTHON implementation for the ontology reasoner.

- dataReader2016.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms.

- getBERTusingColab.py, prepareBERT.py, prepareELMo.py: files used to extract the BERT, respectively, ELMo word embeddings and prepare the final BERT, respectively, ELMo embedding matrix, training, and testing data sets. 
