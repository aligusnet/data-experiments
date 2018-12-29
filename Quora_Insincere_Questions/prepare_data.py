import os
import numpy as np
import pandas as pd
import spacy
import time
from tqdm import tqdm
from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence


def logfunc(func):
    "Decorator that logs the geven function's runnng time"

    def logged(*args):
        print('===> running', func.__name__, '...')
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        print('<=== finished {} in {:03.2f} s.'.format(func.__name__, elapsed))
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
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


    def prepare_data(self):
        quora_df = pd.read_csv(self._get_data_path('train.csv'), nrows = self.nrows)
        clean_data_filepath = self._clean(quora_df)
        bigram_model_filepath, trigram_model_filepath, sentences_filepath = self._build_phrase_models(clean_data_filepath)
        bigram_model = Phrases.load(bigram_model_filepath)
        trigram_model = Phrases.load(trigram_model_filepath)


    @logfunc
    def _clean(self, quora_df):
        result_filepath = self._get_output_path('clean_data.txt')
        if not self.force and os.path.isfile(result_filepath):
            return result_filepath
        
        cleaned_data = [self._clean_question_text(text) for text in self.nlp.pipe(tqdm(quora_df.question_text))]
        with open(result_filepath, 'w', encoding='UTF-8') as f:
            f.write('\n'.join(cleaned_data) + '\n')
        
        return result_filepath


    def _clean_question_text(self, doc):
        sents = []
        for sent in doc.sents:
            cleaned_sent = ' '.join([token.lemma_ for token in sent if not self._punct_space(token)])
            sents.append(cleaned_sent)

        return '. '.join(sents)


    def _punct_space(self, token):
        return token.is_punct or token.is_space


    @logfunc
    def _build_phrase_models(self, clean_data_filepath):
        "Phrase Modeling"

        bigram_sentences_filepath = self._get_output_path('bigram_sentences.txt')
        trigram_sentences_filepath = self._get_output_path('trigram_sentences.txt')
        bigram_model_filepath = self._get_output_path('bigram.model')
        trigram_model_filepath = self._get_output_path('trigram.model')

        unigram_sentences = LineSentence(clean_data_filepath)
        bigram_model = Phrases(unigram_sentences)
        bigram_model.save(bigram_model_filepath)
        self._save_sentences(unigram_sentences, bigram_model, bigram_sentences_filepath)
        
        bigram_sentences = LineSentence(bigram_sentences_filepath)
        trigram_model = Phrases(bigram_sentences)
        trigram_model.save(trigram_model_filepath)
        self._save_sentences(bigram_sentences, trigram_model, trigram_sentences_filepath)

        return (bigram_model_filepath, trigram_model_filepath, trigram_sentences_filepath)


    def _save_sentences(self, sents, model, filepath):
        with open(filepath, 'w', encoding='UTF-8') as f:
            for sent in sents:
                new_sent = ' '.join(model[sent])
                f.write(new_sent + '\n')


    def _get_data_path(self, filename):
        return os.path.join(self.data_dir, filename)


    def _get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)
    

if __name__ == '__main__':
    data = Data(10_000, True)
    data.prepare_data()
