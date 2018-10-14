
from inscriptis import get_text
import re
import os
import sys
sys.path.insert(0, '../')

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

def pdf2text_parts(raw_data):
    return {'text': raw_data, 'title': ''}


from bs4 import  BeautifulSoup
from bs4.element import Comment
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def bs4_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    if not bool(soup.find()):
        raise -1

    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

EXTRACT_TITLE = re.compile('<title>(.*?)<\/title>', re.DOTALL)
TEXT_EXTRACTOR_TYPE = 'bs4'
def html2text(raw_html):
    title_search = re.search(EXTRACT_TITLE, raw_html)
    if title_search is not None:
        title = title_search.groups()[0]
    else:
        title = ''

    raw_html = raw_html.replace("<", " <").replace(">", "> ")

    text_inscr = get_text(raw_html)
    text_bs = bs4_text_from_html(raw_html)

    text = ''
    if len(text_inscr) == 0:
        text = text_bs
    else:
        text = text_inscr

    #
    # if TEXT_EXTRACTOR_TYPE in ['inscriptis']:
    #     text = get_text(raw_html)
    # else:
    #     text = bs4_text_from_html(raw_html)

    return {'text': text, 'title': title}

doc_part_names = ['text', 'title']

def get_is_doc_html(text):
    return bool(BeautifulSoup(text, "html.parser").find())


def parse_doc(doc):
    text = doc.text
    if doc.text is None:
        try:
            text = doc.read()
        except:
            print('unable read document: ' + str(doc))
            return None

    if '!DOCTYPE' in text or '<body' in text:
        doc.type = Document.HTML_TYPE


    parts = None
    if doc.type is Document.HTML_TYPE:
        try:
            parts = html2text(text)
        except:
            doc.type = Document.PDF_TYPE
            parts = pdf2text_parts(text)
    elif doc.type is Document.PDF_TYPE:
        parts = pdf2text_parts(text)

    if parts is None:
        return None

    return parts



from tokenization import Tokenizer, Lemmatizer

tokenizer = Tokenizer()
lemmatizer = Lemmatizer()

TITLE_SPLITTER = '\n'

def process_doc(args):
    doc, processed_folder = args
    parts = parse_doc(doc)
    lemmatized_parts = {}
    for p_name in parts.keys():
        tokens = tokenizer.fit_transform(parts[p_name])
        lemm_tokens = lemmatizer.fit_transform(tokens)
        lemmatized_parts[p_name] = lemm_tokens

    file = open(processed_folder + str(doc.id) + '.txt', 'w')
    file.write(' '.join(lemmatized_parts['title']))
    file.write(TITLE_SPLITTER)
    file.write(' '.join(lemmatized_parts['text']))
    file.close()




def get_documents(data_folder):
    urls_dict = read_urls_dict(data_folder + 'urls.numerate.txt')
    docs = read_documents(data_folder)

    documents = []
    errors = []

    for d in docs:
        if d.url not in urls_dict.inverted_dict:
            errors.append(d.url)
            continue

        d.id = urls_dict.get_inverted(d.url)
        documents.append(d)

    return documents, errors


def read_documents(data_folder):
    content_folder = data_folder + 'content/'
    documents = []
    for folder_path in os.listdir(content_folder):
        filepath = content_folder + folder_path + '/'
        for fname in os.listdir(filepath):
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


import pickle
from multiprocessing import Pool

from tqdm import tqdm

if __name__ == "__main__":
    print('started parsing')
    data_folder = 'Data/'
    processed_folder = data_folder + 'text_documents/'


    documents = pickle.load(open(data_folder + 'documents.pkl', 'rb'))#get_documents(data_folder)
    documents = documents.docs
    for d in documents:
        d.data_path = data_folder

    pool = Pool(8)

    tasks = list(zip(documents, [processed_folder] * len(documents)))

    for i in tqdm(pool.imap_unordered(process_doc, tasks), total=len(tasks)):
        continue

    print('all done!')


