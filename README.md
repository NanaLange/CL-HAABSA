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

- main.py: program to run single in-sample and out-of-sample valdition runs. Each method can be activated by setting its corresponding boolean to True e.g. to run the *baby steps* curriculum algorithm, set runBaby_Steps = True.

- main_cross.py: similar to main.py but runs a 10-fold cross validation procedure for each method.

- main_hyper.py: program that performs hyperparameter optimzation for a given space of hyperparamters for each method. To change a method, change the objective and space parameters in the run_a_trial() function.

- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.

- dataReader2016.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms.

- lcrModel.py: Tensorflow implementation for the LCR-Rot algorithm.

- lcrModelAlt.py: Tensorflow implementation for the LCR-Rot-hop algorithm.

- lcrModelInverse.py: Tensorflow implementation for the LCR-Rot-inv algorithm.

- lcrModelAlt_hierarchical_v1, lcrModelAlt_hierarchical_v2, lcrModelAlt_hierarchical_v3, and lcrModelAlt_hierarchical_v4: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention. The number stands for which method for hierarchical attention is used (see HAABSA++ paper). In CL-HAABSA++, we use an altered version of lcrModelAlt_hierarchical_v4, which is discussed next.

- lcrModelAlt_hierarchical_v4_trainevaltest.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4). In contrary to the methods discussed before, this program takes as input not only train and test data, but also validation data. The train data should be split in train (80%) and validation (20%) before running this method. The validation data is needed to decide when the model has converged.

- lcrModelAlt_hierarchical_v4_baby_steps.py and lcrModelAlt_hierarchical_v4_one_pass.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4) with the *baby steps* curriculum strategy implemented. For every bucket, or combination of buckets, data, the optimal hyperparameters can be set in this file. Furthermore, the indices sorted by their curriculum scores has to be added as input. 

- lcrModelAlt_hierarchical_hyper_opt.py: Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention (method 4) that can be used for the hyperparameter optimization of the buckets of training data. 

- auxModel.py: the auxiliary feed-forward model used to compute the curriculum scores of the instances.

- getCurriculumHyperData.py: program that uses the curriculum scores to divide the training data in *k* buckets, for *k*=1,..,10. For every bucket of data, the data is separated in a train and an evaluation set (80-20%) and saved in new files. These files can then be used to optimize the hyperparameters for every bucket of data, using main_hyper.py.

- hyperOpt.py: implementation of HyperOpt to optimize the hyperparameters of the auxiliary feed-forward model used for curriculum learning.

- sentiWordNet.py: file used to determine the SentiWordNet features for every sentence. These features can be fed into auxModel.py to obtain the curriculum scores for the instances. 

- att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.

- cabascModel.py: Tensorflow implementation for the CABASC algorithm

- svmModel.py: PYTHON implementation for a BoW model using a SVM.

- OntologyReasoner.py: PYTHON implementation for the ontology reasoner.

- getBERTusingColab.py, prepareBERT.py, prepareELMo.py: files used to extract the BERT, respectively, ELMo word embeddings and prepare the final BERT, respectively, ELMo embedding matrix, training, and testing data sets. 
