import numpy as np

if not 'logfunc' in globals():
    from logfunc import logfunc


@logfunc
def load_glove(filepath):
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    return dict(get_coefs(*o.split(" ")) for o in open(filepath, encoding = 'UTF-8'))


def _get_vector(index, word):
    vector = index.get(word)

    if vector is None:
        vector = index.get(word.capitalize())

    if vector is None:
        vector = index.get(word.upper())

    if vector is None:
        vector = index.get(word.lower())

    return vector
    

@logfunc
def build_embeddings_matrix(embeddings_index, word_index, max_features):
    embed_mean, embed_std = -0.005838499,0.48782197
    embed_size = 300
    num_words = min(max_features, len(word_index))
    embeddings = np.random.normal(embed_mean, embed_std, (num_words, embed_size))

    for word, index in word_index.items():
        if index >= max_features: continue
        vector = _get_vector(embeddings_index, word)
        if vector is not None: embeddings[index] = vector

    return embeddings


if __name__ == '__main__':
    import os
    import pandas as pd

    data_path = os.path.join('data', '.input')
    glove_filepath = os.path.join(data_path, 'embeddings', 'glove.840B.300d', 'glove.840B.300d.txt')
    quora_path = os.path.join(data_path, 'train.csv')

    word_index = {'': 0, 'hello': 1, 'world': 2, 'there': 3}

    glove = load_glove(glove_filepath)

    embeddings = build_embeddings_matrix(glove, word_index, 10)
    print(embeddings)
