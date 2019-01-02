"mitigate data skew issue"

import os
import pandas as pd
import numpy as np


def balance_data(quora_df, multiplicator = 1, random_state = None):
    np.random.seed(random_state)

    print('before balancing, total cases:', quora_df.shape[0])
    positive_df = quora_df.loc[quora_df.target == 1]
    negative_df = quora_df.loc[quora_df.target == 0]

    print('positive cases:', positive_df.shape[0], ', negative cases:', negative_df.shape[0])

    positive_nrows = positive_df.shape[0]
    negative_nrows = negative_df.shape[0]

    indices = np.random.choice(negative_nrows, positive_nrows * multiplicator, replace = False)
    negative_df = negative_df.values[indices]
    data = np.vstack([positive_df.values, negative_df])
    np.random.shuffle(data)
    df = pd.DataFrame(data=data, columns = quora_df.columns)

    print('after balancing, total cases:', df.shape[0])
    positive_df = df.loc[df.target == 1]
    negative_df = df.loc[df.target == 0]
    print('positive cases:', positive_df.shape[0], ', negative cases:', negative_df.shape[0])

    return df


if __name__ == '__main__':
    input_path = os.path.join('data', '.input', 'train.csv')
    output_path = os.path.join('data', '.input', 'balanced_train_4.csv')

    quora_df = pd.read_csv(input_path)
    balanced_df = balance_data(quora_df, multiplicator = 4)
    balanced_df.to_csv(output_path, index = False)

