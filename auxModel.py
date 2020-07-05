import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
import tensorflow as tf


def main(tr_features, te_features, tr_sent, te_sent):

    model = Sequential()
    model.add(Dense(183, input_dim=25, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(140, activation='relu', kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.keras.initializers.Zeros()))
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(tr_features, tr_sent, epochs=600, batch_size=198, verbose=0)

    trainy = model.predict(tr_features)
    tr_pred = list()
    for i in range(len(trainy)):
        tr_pred.append(np.argmax(trainy[i]))
    train = list()
    for i in range(len(tr_sent)):
        train.append(np.argmax(tr_sent[i]))
    tr_acc = accuracy_score(tr_pred, train)
    print('Train accuracy is: ', tr_acc * 100)

    y_pred = model.predict(te_features)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    test = list()
    for i in range(len(te_sent)):
        test.append(np.argmax(te_sent[i]))
    a = accuracy_score(pred, test)
    print('Test accuracy is: ', a * 100)

    curr_scores = []
    for t, p in zip(trainy, tr_sent):
        score = np.sum(np.square(np.subtract(t, p)))
        curr_scores.append(score)

    return np.asarray(curr_scores)