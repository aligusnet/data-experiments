import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


class Metrics(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data


    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0])).round()
        target = self.validation_data[1]

        tn, fp, fn, tp = confusion_matrix(target, predict).ravel()

        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn)  if (tp + fn) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0.0 else 0.0

        print('Epoch {}/{}'.format(epoch + 1, self.params['epochs']))
        print('[Valuation] f1: {:.4f} — precision: {:.4f} — recall: {:.4f} - accuracy: {:.4f} '.format(f1, precision, recall, accuracy))
        print('[Confusion Matrix] [tn, fp, fn, tp] [{}, {}, {}, {}]'.format(tn, fp, fn, tp))
