import os
import subprocess
import numpy as np
from uuid import uuid1
from sklearn.base import BaseEstimator

class ExecutionError(Exception):
    def __init__(self, message, proc):
        self.message = message
        self.returncode = proc.returncode
        self.args = proc.args
        self.output = proc.stderr

    def __str__(self):
        msg = '{0}{1}{1}'.format(self.message, os.linesep)
        msg += 'Return code: {0}{1}'.format(self.returncode, os.linesep)
        msg += 'Arguments: {0}{1}'.format(self.args, os.linesep)
        msg += 'Command line: {0}{1}'.format(' '.join(self.args), os.linesep)
        msg += 'Output:{0}'.format(os.linesep)
        msg += self.output
        return msg


class VowpalWabbitClassifier(BaseEstimator):
    def __init__(self, vw_path = 'vw', working_dir = '.', fit_params = None, debug = False):
        self._vw = vw_path
        self._working_dir = working_dir
        self._fit_params = {} if fit_params is None else fit_params
        uuid = str(uuid1())
        self._model = os.path.join(self._working_dir, uuid+'.model')
        self._predictions = os.path.join(self._working_dir, uuid+'_predictions.txt')
        self._debug = debug

        if not os.path.exists(self._working_dir):
            os.makedirs(self._working_dir)

    def fit(self, path_to_data, labels=None):
        args = [self._vw, 
                '-d', path_to_data, 
                '-f', self._model]
        for k, v in self._fit_params.items():
            args.append(k)
            args.append(str(v))
        proc = subprocess.run(args, capture_output=True, text=True)
        self.fit_output_ = proc.stderr
        if proc.returncode != 0:
            raise ExecutionError('fitting failed', proc)
        if self._debug:
            print(self.fit_output_)
        return self

    def _predict(self, path_to_data):
        args = [self._vw, 
                '-t', '-d', path_to_data, 
                '-i', self._model, 
                '-p', self._predictions]
        proc = subprocess.run(args, capture_output=True, text=True)
        self.transform_output_ = proc.stderr
        if proc.returncode != 0:
            raise ExecutionError('transform failed', proc)

        self.predict_proba_ = np.fromfile(self._predictions, dtype = float, sep = os.linesep)

        to_predictions = np.vectorize(lambda x: -1.0 if x < 0.0 else 1.0)
        self.predictions_ = to_predictions(self.predict_proba_)

        if self._debug:
            print(self.transform_output_)
    
    def predict(self, path_to_data):
        self._predict(path_to_data)

        return self.predictions_

    def predict_proba(self, path_to_data):
        self._predict(path_to_data)

        return self.predict_proba_


if __name__ == '__main__':
    fit_params = {
        '--loss_function': 'logistic',
        '-b': 27,
    }
    try:
        vw = VowpalWabbitClassifier(working_dir = '.input', fit_params = fit_params, debug = True)
        vw.fit('Quora_Insincere_Questions/Vowpal_Wabbit/.input/train.vw')
        vw.predict('Quora_Insincere_Questions/Vowpal_Wabbit/.input/test.vw')
        print(vw.predictions_[45:55])
        print()
        print(vw.predict_proba_[45:55])
    except ExecutionError as e:
        print(e)
