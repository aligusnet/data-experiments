import os
import numpy as np
import pandas as pd
import spacy
import time
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence


def logfunc(func):
    "Decorator that logs the geven function's runnng time"

    def logged(*args, **kwargs):
        print('===> running', func.__name__, '...')
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print('<=== finished {} in {:03.2f} s.'.format(func.__name__, elapsed))
        return result
    return logged


class Data:
    def __init__(self, output_dir, force):
        self.force = force
        self.output_dir = output_dir
    
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        
        self._load_nlp()


    @logfunc
    def _load_nlp(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


    def fit(self, quora_df):
        clean_data_filepath = self._clean(quora_df)
        sentences_filepath = self._build_phrase_models(clean_data_filepath)

        return self._build_df(quora_df, sentences_filepath)

    def transform(self, quora_df):
        self.force = True
        clean_data_filepath = self._clean(quora_df)
        sentences_path = self._transform_with_phrase_models(clean_data_filepath)
        return self._build_df(quora_df, sentences_path)


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

        cleaned = '. '.join(sents)
        if len(cleaned) == 0:
            cleaned = '-EMPTY-'
        return cleaned


    def _punct_space(self, token):
        return token.is_punct or token.is_space


    @logfunc
    def _build_phrase_models(self, clean_data_filepath):
        "Phrase Modeling"

        bigram_sentences_filepath = self._get_output_path('bigram_sentences.txt')
        trigram_sentences_filepath = self._get_output_path('trigram_sentences.txt')

        self.bigram_model = self._train_phrase_detection_model(clean_data_filepath, bigram_sentences_filepath)
        self.trigram_model = self._train_phrase_detection_model(bigram_sentences_filepath, trigram_sentences_filepath)

        return trigram_sentences_filepath


    def _train_phrase_detection_model(self, input_filepath, output_filepath):
        sentences = LineSentence(input_filepath)
        model = Phraser(Phrases(sentences))
        self._save_sentences(sentences, model, output_filepath)
        return model


    def _save_sentences(self, sents, model, filepath):
        with open(filepath, 'w', encoding='UTF-8') as f:
            for sent in sents:
                new_sent = ' '.join(model[sent])
                f.write(new_sent + '\n')


    def _transform_with_phrase_models(self, clean_data_filepath):
        bigram_sentences_filepath = self._get_output_path('bigram_sentences2.txt')
        trigram_sentences_filepath = self._get_output_path('trigram_sentences2.txt')

        self._save_sentences(LineSentence(clean_data_filepath), self.bigram_model, bigram_sentences_filepath)
        self._save_sentences(LineSentence(bigram_sentences_filepath), self.trigram_model, trigram_sentences_filepath)

        return trigram_sentences_filepath


    @logfunc
    def _build_df(self, quora_df, sentences_filepath):
        nrows = quora_df.shape[0]
        sentences = self._read_sentences(sentences_filepath)
        if len(sentences) != nrows:
            print('Expected number of sentences is ', nrows, 
                  ' but read ', len(sentences),
                  ' must rebuild sentences')
            raise Exception('Unexpected number of sentences read')
        
        data = {'qid': quora_df.qid, 'question_text': sentences}
        if 'target' in quora_df:
            data['target'] = quora_df.target
        return pd.DataFrame(data)


    def _read_sentences(self, sentences_filepath):
        with open(sentences_filepath, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        return list(map(lambda line: line.rstrip(), lines))


    def _get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)


if __name__ == '__main__':
    nrows = None
    data_path = os.path.join('data', '.input', 'balanced_train_4.csv')
    output_dir = os.path.join('data', '.input', 'full' if nrows is None else str(nrows))
    result_filepath = os.path.join(output_dir,'preprocessed.csv')

    quora_df = pd.read_csv(data_path, nrows = nrows)
    data = Data(output_dir, False)

    df = data.fit(quora_df)
    df.to_csv(result_filepath, index = False)

    print(data.transform(quora_df[:100]))
