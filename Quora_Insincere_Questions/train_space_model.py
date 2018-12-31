import pandas as pd
import numpy as np
import os
import spacy
from spacy.util import minibatch, compounding
from collections import namedtuple


class Classifier:
    LABEL = 'POSITIVE'
    def __init__(self, threshold = 0.5):
        self.nlp = spacy.blank('en')
        self.threshold = threshold
        self.textcat = Classifier._get_textcat(self.nlp)
        self.output_dir = os.path.join('data', '.input')
        self.optimizer = None

    @staticmethod
    def _get_textcat(nlp):
        if 'textcat' not in nlp.pipe_names:
            textcat = nlp.create_pipe('textcat')
            nlp.add_pipe(textcat, last=True)
        else:
            textcat = nlp.get_pipe('textcat')
        textcat.add_label(Classifier.LABEL)
        return textcat


    @staticmethod
    def convert_data(quora_df):
        annotations = quora_df.target.apply(lambda v: {'cats': {Classifier.LABEL: bool(int(v))}})
        df = pd.DataFrame({'texts': quora_df.question_text, 'annotations': annotations})
        return df


    def train(self, train_df, test_df, n_iter=1000):
        nlp = self.nlp
        textcat = self.textcat
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
        with nlp.disable_pipes(*other_pipes):
            self.optimizer = nlp.begin_training()
            self._print_optimizer_params(self.optimizer)
            print("Training the model...")
            Stats.print_header()
            batch_size = compounding(40., 1000., 1.001)
            losses = {}
            for i in range(n_iter):
                # batch up the examples using spaCy's minibatch
                batch = Classifier._get_batch(train_df, next(batch_size))
                nlp.update(batch.texts, batch.annotations, sgd=self.optimizer, drop=0.2, losses=losses)

                if i % 50 == 0:
                        scores = self._evaluate(test_df)
                        Stats.print_scores(losses, scores)
        nlp.to_disk(self.output_dir)


    def predict(self, texts):
        with self.textcat.model.use_params(self.optimizer.averages):
            docs = (self.nlp.tokenizer(text) for text in texts)
            return self.textcat.predict(list(docs))[0]


    @staticmethod
    def _print_optimizer_params(optimizer):
        print('Optimizer params:')
        for k, v in optimizer.__dict__.items():
            print('\t{}:'.format(k), v)


    @staticmethod
    def _get_batch(df, batch_size):
        nrows = df.shape[0]
        row_numbers = np.random.choice(nrows, int(batch_size), replace = False)
        return df.iloc[row_numbers]


    def _evaluate(self, df):
        golds = get_scores(df.annotations.values, Classifier.LABEL)
        
        preds = self.predict(df.texts)

        tp = 0   # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 1   # True negatives
        for i in range(len(golds)):
            if preds[i] >= self.threshold and golds[i] >= self.threshold:
                tp += 1
            elif preds[i] >= self.threshold and golds[i] < self.threshold:
                fp += 1
            elif preds[i] < self.threshold and golds[i] < self.threshold:
                tn += 1
            elif preds[i] < self.threshold and golds[i] >= self.threshold:
                fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall != 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0
        return Scores(precision, recall, f_score, tp, fp, fn, tn)


def get_scores(annotations, cat_name):
    return v_get_score_values(annotations, cat_name)


def get_score_values(annotation, cat_name):
    return float(annotation['cats'][cat_name])

v_get_score_values = np.vectorize(get_score_values, otypes=[float])


def score_to_pred(score, threshold):
    return 0 if score < threshold else 1

v_score_to_pred = np.vectorize(score_to_pred, otypes=[int])


Scores = namedtuple('scores', ['precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'])


class Stats:
    @staticmethod
    def print_header():
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{}'.format('LOSS', 'P', 'R', 'F1', 'Confusion Matrix'))
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{}'.format('', '', '', '', '[TP, FP, FN, TN]'))

    @staticmethod
    def print_scores(losses, scores):
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4}'  # print a simple table
                .format(losses['textcat'], scores.precision,
                        scores.recall, scores.f1,
                        [int(scores.tp), int(scores.fp), int(scores.fn), int(scores.tn)]))



if __name__ == '__main__':
    data_path = os.path.join('data', '.input', 'full', 'preprocessed.csv')
    quora_df = Classifier.convert_data(pd.read_csv(data_path))
    nrows = quora_df.shape[0]
    train_nrows = nrows - nrows // 50
    train_df = quora_df.iloc[:train_nrows]
    test_df = quora_df.iloc[train_nrows:]

    clf = Classifier()
    clf.train(train_df, test_df.iloc[:200])
