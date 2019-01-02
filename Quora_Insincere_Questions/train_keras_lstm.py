import numpy as np
import pandas as pd
import spacy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from collections import namedtuple
import os
from tqdm import tqdm

from prepare_data import logfunc


LstmShape = namedtuple('LstmShape', 'num_hidden max_length num_class'.split())
LstmSettings = namedtuple('LstmSettings', 'dropout learn_rate'.split())


class LstmClassifier:
    def __init__(self, shape, settings):
        self.nlp = self._load_nlp()
        self.shape = shape
        self.settings = settings


    @staticmethod
    @logfunc
    def _load_nlp():
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        return nlp


    @logfunc
    def train(self, train_df, val_df, batch_size=100, nb_epoch=5):
        train_X = self._get_features(train_df.question_text)
        val_X = self._get_features(val_df.question_text)

        model = self._compile_model(self._get_embedings(), self.shape, self.settings)
        model.fit(train_X, train_df.target, validation_data=(val_X, val_df.target),
            epochs=nb_epoch, batch_size=batch_size)
        self.model = model


    def _get_embedings(self):
        return self.nlp.vocab.vectors.data


    @logfunc
    def _get_features(self, docs):
        Xs = np.zeros((len(docs), self.shape.max_length), dtype='int32')
        docs = self.nlp.pipe(docs)
        for i, doc in enumerate(tqdm(docs)):
            for j, token in enumerate(doc[:self.shape.max_length]):
                Xs[i, j] = self._get_vector(token)
        return Xs


    def _get_vector(self, token):
        vector_id = token.vocab.vectors.find(key=token.orth)
        if vector_id < 0 and '_' in token.orth_:
            doc = self.nlp(token.orth_.replace('_', ' '))
            if doc.vector_norm > 0:
                self.nlp.vocab.set_vector(token.orth, doc.vector)
                vector_id = token.vocab.vectors.find(key=token.orth)
            else:
                print('no vector for', token.orth_)
        return vector_id if vector_id > 0 else 0


    @staticmethod
    @logfunc
    def _compile_model(embeddings, shape, settings):
        model = Sequential()
        model.add(
            Embedding(
                embeddings.shape[0],
                embeddings.shape[1],
                input_length=shape.max_length,
                trainable=False,
                weights=[embeddings],
                mask_zero=True
            )
        )
        model.add(TimeDistributed(Dense(shape.num_hidden, use_bias=False)))
        model.add(Bidirectional(LSTM(shape.num_hidden,
                                    recurrent_dropout=settings.dropout,
                                    dropout=settings.dropout)))
        model.add(Dense(shape.num_class, activation='sigmoid'))
        model.compile(
            optimizer=Adam(lr=settings.learn_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy'])
        return model


if __name__ == '__main__':
    shape = LstmShape(64, 100, 1)
    settings = LstmSettings(0.5, 0.001)

    data_path = os.path.join('data', '.input', 'full', 'preprocessed.csv')
    quora_df = pd.read_csv(data_path, nrows=50_000)
    nrows = quora_df.shape[0]
    train_nrows = nrows - 500
    train_df = quora_df.iloc[:train_nrows]
    val_df = quora_df.iloc[train_nrows:]

    lstm = LstmClassifier(shape, settings)
    lstm.train(train_df, val_df)
