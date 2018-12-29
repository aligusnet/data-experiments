import os
import numpy as np
import pandas as pd
import spacy
import time
from tqdm import tqdm


def logfunc(func):
    def logged(*args):
        print('running', func.__name__, '...')
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        print('finished {} in {:03.2f} s.'.format(func.__name__, elapsed))
        return result
    return logged

class Data:
    def __init__(self, nrows, force):
        self.nrows = nrows
        self.force = force
        self.data_dir =  os.path.join('data', '.input')
        if self.nrows:
            self.output_dir = os.path.join(self.data_dir, 'nrows_' + str(self.nrows))
        else:
            self.output_dir = os.path.join(self.data_dir, 'full')
    
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        
        self._load_nlp()

    @logfunc
    def _load_nlp(self):
        self.nlp = spacy.load('en_core_web_sm')

    def prepare_data(self):
        quora_df = pd.read_csv(self._get_data_path('train.csv'), nrows = self.nrows)
        clean_data_filepath = self._clean(quora_df)


    @logfunc
    def _clean(self, quora_df):
        result_filepath = self._get_output_path('clean_data.txt')
        if not self.force and os.path.isfile(result_filepath):
            return result_filepath
        
        cleaned_data = [self._clean_question_text(text) for text in tqdm(quora_df.question_text)]
        with open(result_filepath, 'w', encoding='UTF-8') as f:
            f.write('\n'.join(cleaned_data) + '\n')
        
        return result_filepath

    def _clean_question_text(self, text):
        doc = self.nlp(text)
        sents = []
        for sent in doc.sents:
            cleaned_sent = ' '.join([token.lemma_ for token in sent if not self._punct_space(token)])
            sents.append(cleaned_sent)

        return '. '.join(sents)


    def _punct_space(self, token):
        return token.is_punct or token.is_space


    def _get_data_path(self, filename):
        return os.path.join(self.data_dir, filename)


    def _get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)
    

if __name__ == '__main__':
    data = Data(10_000, False)
    data.prepare_data()
