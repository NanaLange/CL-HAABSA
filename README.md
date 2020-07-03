# CL-HAABSA
Implementation of curriculum learning (CL) in HAABSA++

Code for implementing curriculum learning in the two-step Hybrid Approach for Aspect-Based Sentiment Analysis (HAABSA) with contextual word embeddings and hierarchical attention (HAABSA++).
The HAABSA++ paper can be found via: https://arxiv.org/pdf/2004.08673.pdf.
The HAABSA paper can be found via: https://personal.eur.nl/frasincar/papers/ESWC2019/eswc2019.pdf.

The HAABSA++ model uses a domain sentiment ontology and a neural network as backup. Curriculum learning (CL) is used to improve the results of the HAABSA++ model.


## Software installation
First, the right environment needs to be set up and the right files need to be downloaded. This can be done by following the installation instructions given at https://github.com/ofwallaart/HAABSA. Only the files mentioned in the Read Me have to be downloaded, there is no need to download the HAABSA files. Hereafter, the right word embeddings need to be downloaded via: https://github.com/mtrusca/HAABSA_PLUS_PLUS. Again, only the files mentioned in the Read Me should be downloaded. 

After completing the instructions, all the CL-HAABSA files need to be installed into the newly created environment. An explanation of the CL-HAABSA files is given below.


## Software explanation
The are three main files in the environment that can be run: main.py, main_cross.py, and main_hyper.py. An overview of these files and other important files is given:

- main.py: program to run single in-sample and out-of-sample valdition runs. Each method can be activated by setting its corresponding boolean to True e.g. to run the *baby steps* curriculum algorithm, set runBaby_Steps = True.

- main_cross.py: similar to main.py but runs a 10-fold cross validation procedure for each method.

- main_hyper.py: program that performs hyperparameter optimzation for a given space of hyperparamters for each method. To change a method, change the objective and space parameters in the run_a_trial() function.

- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.

- dataReader2016.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms

- lcrModel.py: Tensorflow implementation for the LCR-Rot algorithm

- lcrModelAlt.py: Tensorflow implementation for the LCR-Rot-hop algorithm

- lcrModelInverse.py: Tensorflow implementation for the LCR-Rot-inv algorithm

- lcrModelAlt_hierarchical_v1, lcrModelAlt_hierarchical_v2 (second method), lcrModelAlt_hierarchical_v3 (third method), and lcrModelAlt_hierarchical_v4 (fourth method): Tensorflow implementation for the LCR-Rot-hop algorithm with hierarchical attention. The number stands for which method for hierarchical attention is used (see HAABSA++ paper). In CL-HAABSA++, we use method 4, so lcrModelAlt_hierarchical_v4.

cabascModel.py: Tensorflow implementation for the CABASC algorithm

OntologyReasoner.py: PYTHON implementation for the ontology reasoner

svmModel.py: PYTHON implementation for a BoW model using a SVM.

att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.
