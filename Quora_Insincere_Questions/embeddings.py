import numpy as np

if not 'logfunc' in globals():
    from logfunc import logfunc


@logfunc
def load_glove(filepath):
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    return dict(get_coefs(*o.split(" ")) for o in open(filepath, encoding = 'UTF-8'))


if __name__ == '__main__':
    import os
    import pandas as pd

    data_path = os.path.join('data', '.input')
    glove_filepath = os.path.join(data_path, 'embeddings', 'glove.840B.300d', 'glove.840B.300d.txt')
    quora_path = os.path.join(data_path, 'train.csv')

    embeddings = load_glove(glove_filepath)
