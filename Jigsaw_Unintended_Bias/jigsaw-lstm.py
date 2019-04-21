import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler


Config = namedtuple('Config', 'data_root wv_root max_features max_len'.split())

DEBUG = False

if DEBUG:
    NUM_MODELS = 2
    BATCH_SIZE = 512
    LSTM_UNITS = 64
    DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
    EPOCHS = 2
    NUM_ROWS = 50_000
else:
    NUM_MODELS = 2
    BATCH_SIZE = 512
    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
    EPOCHS = 4
    NUM_ROWS = None

EMBEDDING_FILES = [
    'crawl-300d-2M.wv'
]

def main():
    config = Config(
        max_features = 50_000, 
        max_len = 100,
        data_root = Path('..', 'input', 'jigsaw-unintended-bias-in-toxicity-classification'),
        wv_root = Path('..', 'input', 'gensim-word-vectors')
        )

    h = Helper(config)

    train_df = pd.read_csv(Path(config.data_root, 'train.csv'), nrows=NUM_ROWS)
    test_df = pd.read_csv(Path(config.data_root, 'test.csv'))

    x_train = train_df.comment_text
    y_train = np.where(train_df.target >= 0.5, 1, 0)
    x_test = test_df.comment_text

    y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

    filters = ''.join(set("/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' + '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'))
    tokenizer = text.Tokenizer(num_words = config.max_features, filters=filters)
    tokenizer.fit_on_texts(pd.concat([x_train, x_test]))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=config.max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=config.max_len)

    embedding_matrix = np.concatenate(
        [h.build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
        
    checkpoint_predictions = []
    weights = []

    for model_idx in range(NUM_MODELS):
        model = h.build_model(embedding_matrix, y_aux_train.shape[-1])
        for global_epoch in range(EPOCHS):
            model.fit(
                x_train,
                [y_train, y_aux_train],
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=2,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
                ]
            )
            checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
            weights.append(2 ** global_epoch)

    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

    submission = pd.DataFrame.from_dict({
        'id': test_df['id'],
        'prediction': predictions
    })

    submission.to_csv('submission.csv', index=False)


class Helper:
    def __init__(self, config):
        self.config = config

    def build_matrix(self, word_index, embeddings_filename):
        embedding_index = self.load_embeddings(embeddings_filename)
        emdeb_size = embedding_index['hi'].shape[0]
        embedding_matrix = np.zeros((len(word_index) + 1, emdeb_size))
        for word, index in word_index.items():
            try:
                embedding_matrix[index] = embedding_index[word]
            except KeyError:
                pass
        return embedding_matrix

    def load_embeddings(self, embeddings_filename):
        return KeyedVectors.load_word2vec_format(Path(self.config.wv_root, embeddings_filename), binary = True)

    def build_model(self, embedding_matrix, num_aux_targets):
        words = Input(shape=(self.config.max_len,))
        x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        result = Dense(1, activation='sigmoid')(hidden)
        aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
        
        model = Model(inputs=words, outputs=[result, aux_result])
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model

main()
