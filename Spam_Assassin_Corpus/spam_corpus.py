import os
import re
import shutil
import tarfile
import email
import itertools
import bs4
from email import policy
import pandas as pd
from urllib.parse import urljoin
from urllib.request import urlopen


def read_spam_corpus(working_dir = '.input', force_rebuild = False):
    df_path = os.path.join(working_dir, 'corpus.df')
    if not os.path.exists(df_path) or force_rebuild:
        cr = CorpusReader(working_dir)
        df = cr.read()
        df.to_pickle(df_path)
    else:
        df = pd.read_pickle(df_path)
    return df


class CorpusReader:
    DOCTYPE_REGEX = re.compile(r"^\s*<!DOCTYPE.*?>")
    URL = 'https://spamassassin.apache.org/old/publiccorpus/'
    DATASETS = [
        ('20021010_easy_ham.tar.bz2', 0),
        ('20021010_hard_ham.tar.bz2', 0),
        ('20030228_easy_ham.tar.bz2', 0),
        ('20030228_easy_ham_2.tar.bz2', 0),
        ('20030228_hard_ham.tar.bz2', 0),
        ('20021010_spam.tar.bz2', 1),
        ('20030228_spam.tar.bz2', 1),
        ('20030228_spam_2.tar.bz2', 1),
        ('20050311_spam_2.tar.bz2', 1),
    ]


    def __init__(self, working_dir):
        loader = FolderDownloader(CorpusReader.URL, working_dir)
        self._reader = ArchivedEmailReader(loader)

    def read(self):
        dfs = []
        for filename, label in CorpusReader.DATASETS:
            dfs.append(self._read_to_data_frame(filename, label))
        return pd.concat(dfs)

    def _read_to_data_frame(self, filename, label):
        print('reading {}...'.format(filename), end='', flush=True)
        texts = self._read_texts(filename)
        labels = pd.Series(list(itertools.repeat(label, len(texts))))
        d = {
            'text': texts,
            'label': labels
        }
        print('  done. Read {} records'.format(len(texts)))
        return pd.DataFrame(d)

    def _read_texts(self, filename):
        emails = self._reader.read(filename)
        texts = map(self._extract_text, emails)
        return pd.Series(data = texts)

    def _extract_text(self, msg):
        try:
            body = msg.get_body()
            if not body['content-type']:
                return body.get_content()
            if body['content-type'].maintype == 'text':
                if body['content-type'].subtype == 'plain':
                    return body.get_content()
                elif body['content-type'].subtype == 'html':
                    content = CorpusReader.DOCTYPE_REGEX.sub("", body.get_content())
                    soup = bs4.BeautifulSoup(content, features="html.parser")
                    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
                    return soup.get_text()
        except:
            pass
        return None


class ArchivedEmailReader:
    def __init__(self, loader):
        self._loader = loader

    def read(self, filename):
        path = self._loader.get_file(filename)
        files = self._read_archived_files(path)
        parser = email.parser.BytesParser(policy = policy.default)
        emails = map(parser.parsebytes, files)
        return emails
    
    def _read_archived_files(self, path):
        files = []
        with tarfile.open(path, 'r') as tar:
            for ti in tar:
                if ti.isfile():
                    files.append(tar.extractfile(ti).read())
        return files


class FolderDownloader:
    def __init__(self, folder_url, local_path):
        self._folder_url = folder_url

        self._local_path = local_path
        os.makedirs(self._local_path, exist_ok=True)

    def get_file(self, filename):
        local_path = os.path.join(self._local_path, filename)
        if not os.path.exists(local_path):
            self._download(filename, local_path)
        return local_path

    def _download(self, filename, local_path):
        url = urljoin(self._folder_url, filename)
        with urlopen(url) as response:
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response, f)


if __name__ == '__main__':
    print(read_spam_corpus())
