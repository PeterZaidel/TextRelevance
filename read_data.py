import os
import numpy as np
from Levenshtein import distance as lev_distance
from utils import TwoWayDict





class Document:

    HTML_TYPE = 'html'
    PDF_TYPE = 'pdf'

    def __init__(self, data_path, content_path, filename):
        self.filename = filename
        self.data_path = data_path
        self.content_path = content_path

        self.text = None
        self.id = None
        self.url = None
        self.type = None

        pass

    @staticmethod
    def __split_raw_text(text):
        idx = 0
        while text[idx] in ['\n', '\r', '\t', '\n\t', '\r\n']:
            idx += 1
        text = text[idx:]

        bidx = text.find('\n')

        url_text = text[0: bidx]
        body = text[bidx:]
        return url_text, body


    def get_type(self, url, text):
        if url.find('.pdf') > 0 :
            return self.PDF_TYPE
        else:
            return self.HTML_TYPE


    def extract_url(self, save=False):
        filepath = self.data_path + self.content_path + self.filename
        text = open(filepath, 'r', errors='ignore').read()
        url_text, body = self.__split_raw_text(text)
        self.url = url_text
        if save:
            self.text = body

        self.type = self.get_type(url_text, body)

        return url_text

    def read(self, save=False):
        if self.text is not None:
            return self.text

        filepath = self.data_path + self.content_path + self.filename
        text = open(filepath, 'r', errors='ignore').read()
        url_text, body = self.__split_raw_text(text)

        self.type = self.get_type(url_text, body)
        if save:
            self.text = body
        return body

    def __str__(self):
        return  'doc_id: {0}, doc_url:{1}, doc_type: {2}, doc_filename: {3}'.format(self.id,
                                                                                    self.url,
                                                                                    self.type,
                                                                                    self.filename)


class Documents:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.docs_ids = dict()
        self.docs_urls = dict()
        self.docs = []

    def set_data_folder(self, data_folder):
        self.data_folder = data_folder
        for d in self.docs:
            d.data_folder = data_folder

    def add(self, doc):
        self.docs_ids[doc.id] = doc
        self.docs_urls[doc.url] = doc
        self.docs.append(doc)

    def get_by_id(self, doc_id):
        return self.docs_ids[doc_id]

    def get_by_url(self, doc_url):
        return self.docs_urls[doc_url]

    def __getitem__(self, idx):
        return self.docs[idx]

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        return self.docs



from tokenization import Tokenizer, Lemmatizer
class Queries:
    def __init__(self, ngram_range = None):
        self.queries = None
        self.lemmatized_queries = None
        self.ngram_range = ngram_range

    def lemmatize(self, stop_words=None):
        tokenizer = Tokenizer(stop_words = stop_words)
        lemmatizer = Lemmatizer(stop_words = stop_words)

        self.lemmatized_queries = dict()
        for q_id in self.queries.dict.keys():
            q = self.queries.get(q_id)

            tok_q = tokenizer.fit_transform(q)
            lem_q = lemmatizer.fit_transform(tok_q)
            self.lemmatized_queries[int(q_id)] = lem_q

    def get_ngrams(self, tokens, ngram):
        grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
        return grams

    def create_vocab_single(self):
        words = [w for q_id in self.lemmatized_queries.keys() for w in self.lemmatized_queries[q_id]]
        self.vocab = TwoWayDict()
        idx = 0
        for w in words:
            if self.vocab.dict.get(w, None) is not None:
                continue

            self.vocab.add(w, idx)
            idx += 1

    def create_vocab(self):
        self.create_vocab_single()

        # words = []
        # for ngram in range(self.ngram_range[0], self.ngram_range[1]):
        #     words += [w for q_id in self.lemmatized_queries.keys() for w in self.get_ngrams(self.lemmatized_queries[q_id],
        #                                                                                ngram)]
        # self.vocab = TwoWayDict()
        # idx = 0
        # for w in words:
        #     if self.vocab.dict.get(w, None) is not None:
        #         continue
        #     self.vocab.add(w, idx)
        #     idx += 1

    def load(self, filename):
        self.queries = get_queries(filename)



    def get_token_ids(self):
        res = {}
        for q_id in self.lemmatized_queries.keys():
            q = self.lemmatized_queries[q_id]
            q_tok_ids = []
            if self.ngram_range is not None:
                for ngram in range(self.ngram_range[0], self.ngram_range[1]):
                    try:
                        q_tok_ids += [self.vocab.get(w) for w in self.get_ngrams(q, ngram)]
                    except:
                        print('error: ' + q)
            else:
                q_tok_ids = [self.vocab.get(w) for w in q]

            res[q_id] = q_tok_ids
        return res











def str_replace_all(s, repl):
    new_s = ''
    for a in repl:
        new_s = s.replace(a, '')
    return a


from Levenshtein import distance as lev_distance


def find_most_similar_str(str_list, s):
    distances = []
    for s2 in str_list:
        distances.append(lev_distance(s, s2))

    distances = np.array(distances)
    min_dst_idx = np.argmin(distances)
    return str_list[min_dst_idx], distances[min_dst_idx]


def get_documents(data_folder, save_text=False):
    urls_dict = read_urls_dict(data_folder + 'urls.numerate.txt')
    docs = read_documents(data_folder, save_text)

    documents = Documents(data_folder)
    errors = []

    for d in tqdm(docs):
        if d.url not in urls_dict.inverted_dict:
            errors.append(d.url)
            continue

        d.id = urls_dict.get_inverted(d.url)
        if save_text:
            d.read(save_text)

        documents.add(d)

    return documents

from tqdm import tqdm
def read_documents(data_folder, save_text=False):
    content_folder = data_folder + 'content/'
    documents = []
    for folder_path in tqdm(os.listdir(content_folder)):
        filepath = content_folder + folder_path + '/'
        for fname in tqdm(os.listdir(filepath)):
            doc = Document(data_folder, 'content/' + folder_path + '/', fname)
            doc.extract_url()
            documents.append(doc)

    return documents


def read_urls_dict(filename):
    urls_dict = TwoWayDict()

    file = open(filename, 'r')
    for line in file.readlines():
        data = line.replace('\n', '').split('\t')
        if len(data) == 0:
            continue
        doc_id = int(data[0])
        doc_url = data[1]
        urls_dict.add(doc_id, doc_url)
    return urls_dict


def get_queries(filename):
    file = open(filename, 'r')
    queries = TwoWayDict()
    for line in file.readlines():
        data = line.replace('\n', '').split('\t')
        if len(data) == 0:
            continue
        qid = int(data[0])
        query = data[1]
        queries.add(qid, query)

    return queries