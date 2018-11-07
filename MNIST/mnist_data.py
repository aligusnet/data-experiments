import os
import shutil
import gzip
import numpy as np
from urllib.request import urlopen

class MnistData:
    _SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, local_path = '.input'):
        self._local_path = local_path
        os.makedirs(self._local_path, exist_ok=True)

    def _get_file(self, filename):
        local_path = os.path.join(self._local_path, filename)
        if not os.path.exists(local_path):
            self._download(filename, local_path)
        return local_path

    def _download(self, filename, local_path):
        url = MnistData._SOURCE_URL + filename
        with urlopen(url) as response:
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response, f)

    def _read_int(self, stream):
        dt = np.dtype(np.int32)
        dt = dt.newbyteorder('>')
        return np.frombuffer(stream.read(4), dtype=dt)[0]

    def _extract_features(self, filename):
        MAGIC_NUMBER = 2051
        path = self._get_file(filename)
        with gzip.open(path) as f:
            if self._read_int(f) != MAGIC_NUMBER:
                raise Exception('Incorrect Magic Number in {}'.format(filename))
            nitems = self._read_int(f)
            nrows = self._read_int(f)
            ncols = self._read_int(f)
            print('reading {} items of size {}x{} from {}'.format(nitems, nrows, ncols, filename))
            buffer = f.read(nrows * ncols * nitems)
            data = np.frombuffer(buffer, dtype=np.uint8)
            data = data.reshape(nitems, nrows, ncols, 1)
            data = np.array(data).flatten()
            return data.reshape(nitems, nrows * ncols)

    def _extract_labels(self, filename):
        MAGIC_NUMBER = 2049
        path = self._get_file(filename)
        with gzip.open(path) as f:
            if self._read_int(f) != MAGIC_NUMBER:
                raise Exception('Incorrect Magic Number in {}'.format(filename))
            nitems = self._read_int(f)
            print('reading {} labels from {}'.format(nitems, filename))
            buffer = f.read(nitems)
            data = np.frombuffer(buffer, dtype=np.uint8)
            return data

    def _read_data_set(self, features_filename, labels_filename):
        X = self._extract_features(features_filename)
        y = self._extract_labels(labels_filename)
        return X, y

    def read_train_set(self):
        return self._read_data_set('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

    def read_test_set(self):
        return self._read_data_set('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


if __name__ == '__main__':
    data = MnistData()
    train_X, train_y = data.read_train_set()
    test_X, test_y = data.read_test_set()
    print('train_X', train_X.shape, train_X.dtype)
    print('train_y', train_y.shape, train_y.dtype)
    np.savetxt(".input/train_X.csv", train_X[:100], delimiter=',', fmt='%d')
    np.savetxt(".input/train_y.csv", train_y[:100], delimiter=',', fmt='%d')
