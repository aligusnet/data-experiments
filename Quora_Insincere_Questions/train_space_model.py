import pandas as pd
import numpy as np
import os

import spacy
from spacy.util import minibatch, compounding

class Classifier:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.textcat = Classifier._get_textcat(self.nlp)
        self.output_dir = os.path.join('data', '.input')

    @staticmethod
    def _get_textcat(nlp):
        if 'textcat' not in nlp.pipe_names:
            textcat = nlp.create_pipe('textcat')
            nlp.add_pipe(textcat, last=True)
        else:
            textcat = nlp.get_pipe('textcat')
        textcat.add_label('POSITIVE')
        return textcat


    @staticmethod
    def convert_data(quora_df):
        texts = quora_df.question_text.tolist()
        annotations = [{'POSITIVE': int(y)} for y in quora_df.target.tolist()]
        return (texts, annotations)


    def train(self, train_texts, train_cats, dev_texts, dev_cats, n_iter=20):
        nlp = self.nlp
        textcat = self.textcat
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
        train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            print("Training the model...")
            print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
            for i in range(n_iter):
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = Classifier._evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                    .format(losses['textcat'], scores['textcat_p'],
                            scores['textcat_r'], scores['textcat_f']))
        nlp.to_disk(self.output_dir)


    @staticmethod
    def _evaluate(tokenizer, textcat, texts, cats):
        docs = (tokenizer(text) for text in texts)
        tp = 0.0   # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0   # True negatives
        for i, doc in enumerate(textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall != 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0
        return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    data_path = os.path.join('data', '.input', 'nrows_10000', 'preprocessed.csv')
    quora_df = pd.read_csv(data_path, nrows=10_000)
    nrows = quora_df.shape[0]
    train_nrows = nrows - nrows // 5
    train_df = quora_df.iloc[:train_nrows]
    test_df = quora_df.iloc[train_nrows:]

    train_texts, train_cats = Classifier.convert_data(train_df)
    test_texts, test_cats = Classifier.convert_data(test_df)

    clf = Classifier()
    clf.train(train_texts, train_cats, test_texts, test_cats)
