import keras
import numpy as np
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import accuracy_score
import tensorflow as tf
import utils
from config import *
import sentiWordNet


X_train, tr_sent = sentiWordNet.main(FLAGS.hyper_train_path_ont, FLAGS.hyper_train_aspect_categories)
X_test, te_sent = sentiWordNet.main(FLAGS.hyper_eval_path_ont, FLAGS.hyper_eval_aspect_categories)
y_train = np.asarray(utils.change_y_to_onehot(tr_sent))
y_test = np.asarray(utils.change_y_to_onehot(te_sent))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

space = {'choice': hp.choice('num_layers', [{'layers': 'two', },
                                            {'layers': 'three', 'units3': hp.choice('units3', np.arange(20, 512)),
                                             'dropout3': hp.quniform('dropout3', 0.25, 0.75, 0.1)}]),

         'units1': hp.choice('units1', np.arange(20, 512)),
         'units2': hp.choice('units2', np.arange(20, 512)),

         'dropout1': hp.quniform('dropout1', 0.25, 0.75, 0.1),
         'dropout2': hp.quniform('dropout2', 0.25, 0.75, 0.1),

         'batch_size': hp.choice('batch_size', np.arange(1, 200)),

         'epochs': 200,
         'loss': hp.choice('loss', ['categorical_crossentropy', 'mean_squared_error']),
         'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
         'activation': 'relu'}


def ff_model(hyperparams):
    print('Params testing: ', hyperparams)
    model = Sequential()
    model.add(Dense(hyperparams['units1'], input_dim=X_train.shape[1]))
    model.add(Activation(hyperparams['activation']))
    model.add(Dropout(hyperparams['dropout1']))
    model.add(Dense(hyperparams['units2'], kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                    bias_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation(hyperparams['activation']))
    model.add(Dropout(hyperparams['dropout2']))

    if hyperparams['choice']['layers'] == 'three':
        model.add(Dense(hyperparams['choice']['units3'],
                        kernel_initializer=tf.glorot_uniform_initializer(seed=123),
                        bias_initializer=tf.keras.initializers.Zeros()))
        model.add(Activation(hyperparams['activation']))
        model.add(Dropout(hyperparams['choice']['dropout3']))

    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=hyperparams['learning_rate'])
    model.compile(loss=hyperparams['loss'], optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=0)

    y_pred = model.predict(X_test)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
    acc = accuracy_score(pred, test)
    print("Accuracy is: ", acc)
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(ff_model, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best: ', best)
