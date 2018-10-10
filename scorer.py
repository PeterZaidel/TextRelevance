import sys
sys.path.insert(0, '')

from gen_statistics import  Query, Vocab, get_ngrams, load_queries, load_doc_txt, \
    open_fwd_index, TEXT, TITLE

from metrics import calc_ndcg, save_predict, load_predict

from cachetools import cached


from utils import TwoWayDict
import pickle
import numpy as np

from multiprocessing import Pool

from tqdm import tqdm

from datetime import datetime


def binarize(array, threshold = 0.0):
    return (array > threshold).astype(np.int_)

def fill_non_positive(array, val = 1.0):
    res = array
    res[res <= 0] = val
    return res

class YandexModel:
    def __init__(self, part_names, part_weights, vocab: Vocab,  k1, k2,
                 synonim_weight = 0.7,
                 nmiss_penalty = 0.03,
                 all_words_weight = 0.2,
                 pairs_weight = 0.3,
                 hdr_weight = 0.2):
        self.vocab = vocab
        self.part_names = part_names
        self._part_name_to_idx = dict(zip(part_names, range(len(part_names))))
        self.part_weights = np.array(part_weights)
        self.part_weights_dict = dict(zip(part_names, part_weights))
        self.k1 = k1
        self.k2 = k2
        self.synonim_weight= synonim_weight
        self.nmiss_penalty = nmiss_penalty
        self.all_words_weight=all_words_weight
        self.pairs_weight = pairs_weight
        self.hdr_weight = hdr_weight

    def _gen_icf_unigram(self, counts_unigram, docs_num, tokens_num):
        sum_counts = np.zeros((docs_num, tokens_num))
        for pname in ['text']:
            sum_counts += np.array(counts_unigram[pname])

        icf = sum_counts.sum(0)
        total_count = icf.sum()
        icf[icf == 0] = 1.0
        icf = total_count / icf

        return icf

    def _gen_doc_lens(self, doc_lens, docs_num):
        dl = np.zeros(docs_num)
        for pn in ['text']:
            dl += doc_lens[pn]

        return dl

    def _gen_tf_unigram(self, counts_unigram):
        sum_counts = np.zeros_like(counts_unigram['text'])
        for pname in ['text']:
            sum_counts += np.array(counts_unigram[pname])

        return sum_counts

    def _gen_hdr(self, counts_unigram):

        hdr = np.zeros_like(counts_unigram[TITLE])
        for pname in [TITLE]:
            hdr += binarize(counts_unigram[pname])

        return hdr

    def _gen_tf_bigram(self, counts_bigram):
        return counts_bigram[TEXT].toarray()
        # sum_counts = np.zeros_like(counts_bigram[TEXT])
        # for pname in [TEXT]:
        #     sum_counts += np.array(counts_bigram[pname])
        # return sum_counts

    # def _gen_icf2(self, icf1, pairs_to_idxs):
    #     icf2 = np.zeros((icf1.shape[0], pairs_to_idxs.shape[0]))
    #     icf2 = icf1[pairs_to_idxs[:, 0]] * icf1[pairs_to_idxs[:, 1]]
    #     return icf2
    def _gen_bigrams_to_unigrams(self):
        pairs_to_idxs = np.empty((len(self.vocab.vocab_2), 2), dtype=int)
        for w2 in self.vocab.vocab_2.dict:
            w2_idx = self.vocab.vocab_2[w2]
            w21 = w2[0]
            w22 = w2[1]

            w21_idx = self.vocab.vocab_1[w21]
            w22_idx = self.vocab.vocab_1[w22]

            pairs_to_idxs[w2_idx] = np.array([w21_idx, w22_idx])
        return  pairs_to_idxs


    def fit(self, counts_unigram, counts_bigram_raw,
                                  counts_bigram_inv,
                                   counts_bigram_gap ,
            doc_lens, doc_ids_map: TwoWayDict):

        self.doc_ids_map = doc_ids_map

        docs_num = len(doc_ids_map)

        tokens_num = counts_unigram[self.part_names[0]].shape[1]
        pairs_num = counts_bigram_raw[self.part_names[0]].shape[1]

        self.tf_unigram = self._gen_tf_unigram(counts_unigram)

        self.tf_bigram_raw = self._gen_tf_bigram(counts_bigram_raw)
        self.tf_bigram_inv = self._gen_tf_bigram(counts_bigram_inv)
        self.tf_bigram_gap = self._gen_tf_bigram(counts_bigram_gap)

        self.hdr = self._gen_hdr(counts_unigram)

        self.icf_unigram = self._gen_icf_unigram(counts_unigram, docs_num, tokens_num)
        self.log_icf_unigram = np.log(fill_non_positive(self.icf_unigram))

        self.dl = doc_lens
        self.indxs_bigrams_to_unigrams = self._gen_bigrams_to_unigrams()




        # self.scores_unigram = np.log(icf_unigram) * (self.tf_unigram / (self.tf_unigram + self.k1 + dl[:, None] / self.k2))
        # self.scores_unigram += np.log(icf_unigram) * (0.2 * hdr / (1.0 + hdr))
        #
        # self.log_icf = np.log(icf_unigram)
        # self.bin_tf = tf_unigram
        # self.bin_tf[self.bin_tf > 0] = 1.0
        #
        # self.scores2 = 0.3 * (np.log(icf_unigram[indxs_bigrams_to_unigrams[:, 0]])
        #                       + np.log(icf_unigram[indxs_bigrams_to_unigrams[:, 1]])) * tf_bigram / (1.0 + tf_bigram)

    def unigram_score(self, tokens_idxs, doc_index):
        if tokens_idxs.shape[0] == 0:
            return  0

        tf = self.tf_unigram[doc_index, tokens_idxs]
        hdr = self.hdr[doc_index, tokens_idxs]
        hdr[hdr > 0] = 1.0

        log_icf = self.log_icf_unigram[tokens_idxs]

        score = tf/ (tf + self.k1 + self.dl[doc_index]/ self.k2)
        score += 0.5 * hdr
        score = log_icf * score
        score = score.sum()
        return score





    def bigram_score(self, grams_idxs_list, doc_index):
        # if tokens.shape[0] == 0:
        #     return  0
        #grams, inv_grams, gap_grams = get_ngrams(tokens,ngram=2, inverted=True, with_gap=True)
        # grams_idxs = [self.vocab.vocab_2[g] for g in grams]
        # inv_grams_idxs = [self.vocab.vocab_2[g] for g in inv_grams]
        # gap_grams_idxs = [self.vocab.vocab_2[g] for g in gap_grams]
        grams_idxs, _, _ = grams_idxs_list

        tf = 0
        if grams_idxs.shape[0] > 0:
            tf = 1.5 * self.tf_bigram_raw[doc_index, grams_idxs]
            tf += 0.7 * self.tf_bigram_inv[doc_index, grams_idxs]
            tf += 0.5 * self.tf_bigram_gap[doc_index, grams_idxs]

            sum_icf = self.log_icf_unigram[self.indxs_bigrams_to_unigrams[grams_idxs, 0]] \
                  + self.log_icf_unigram[self.indxs_bigrams_to_unigrams[grams_idxs, 1]]

            score = sum_icf * tf/(1.0 + tf)
            score = score.sum()
            return score
        else:
            return 0.0


        #
        # if grams_idxs.shape[0] > 0:
        #     score_grams = (np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[grams_idxs, 0]])
        #                      + np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[grams_idxs, 1]])) \
        #               * self.tf_bigram[doc_index, grams_idxs] / (1.0 + self.tf_bigram[doc_index, grams_idxs])
        #     score_grams = score_grams.sum()
        # else:
        #     score_grams = 0.0
        #
        # if inv_grams_idxs.shape[0] >  0:
        #     score_inv_grams = (np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[inv_grams_idxs, 0]])
        #                      + np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[inv_grams_idxs, 1]])) \
        #               * self.tf_bigram[doc_index, inv_grams_idxs] / (1.0 + self.tf_bigram[doc_index, inv_grams_idxs])
        #     score_inv_grams = score_inv_grams.sum()
        # else:
        #     score_inv_grams = 0.0
        #
        # if gap_grams_idxs.shape[0] > 0:
        #     score_gap_grams = (np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[gap_grams_idxs, 0]])
        #                      + np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[gap_grams_idxs, 1]])) \
        #               * self.tf_bigram[doc_index, gap_grams_idxs] / (1.0 + self.tf_bigram[doc_index, gap_grams_idxs])
        #
        #     score_gap_grams = score_gap_grams.sum()
        # else:
        #     score_gap_grams = 0.0

        # score = (score_grams + 0.5 * score_inv_grams + 0.1 * score_gap_grams)
        # return  score

    def all_words_score(self, tokens_idxs, doc_index):
        if tokens_idxs.shape[0] == 0:
            return 0
        tf = self.tf_unigram[doc_index, tokens_idxs]
        nmiss = tokens_idxs.shape[0] - np.argwhere(tf > 0).ravel().shape[0]
        score = np.log(self.icf_unigram[tokens_idxs]).sum()
        score *= self.nmiss_penalty ** nmiss
        return score

    def score_query_doc(self, query: Query, doc_id):
        doc_index = self.doc_ids_map[doc_id]
        doc_len = self.dl[doc_index]

        base_tokens = query.tokens
        base_tokens_idxs = query.base_tokens_idxs#np.array([self.vocab.vocab_1[w] for w in base_tokens])

        synonim_tokens = query.synonim_tokens
        synonim_tokens_idxs = query.synonim_tokens_idxs#np.array([self.vocab.vocab_1[w] for w in synonim_tokens])

        score = 0
        score += self.unigram_score(base_tokens_idxs, doc_index)
        score += self.synonim_weight * self.unigram_score(synonim_tokens_idxs, doc_index)

        score += self.pairs_weight * self.bigram_score(query.base_grams_idxs_list, doc_index)
        score += self.pairs_weight * self.synonim_weight * self.bigram_score(query.syn_grams_idxs_list , doc_index)

        #score += self.all_words_weight * self.all_words_score(base_tokens_idxs, doc_index)


        # score += self.pairs_weight * self.bigram_score(base_tokens, doc_index)
        # score += self.pairs_weight * self.synonim_weight * self.bigram_score(synonim_tokens, doc_index)
        #
        # score += self.all_words_weight * self.all_words_score(base_tokens_idxs, doc_index)

        return score


    def _tokens_to_idxs(self, tokens, vocab_dict):
        return np.array([vocab_dict[g] for g in tokens])


    def extend_query(self, query):
        # fill tokens indices
        query.base_tokens_idxs = np.array([self.vocab.vocab_1[w] for w in query.tokens])
        query.synonim_tokens_idxs = np.array([self.vocab.vocab_1[w] for w in query.synonim_tokens])

        # fill base_grams indices for base
        base_grams, base_inv_grams, base_gap_grams = get_ngrams(query.tokens, ngram=2, inverted=True, with_gap=True)
        base_grams_idxs = self._tokens_to_idxs(base_grams, self.vocab.vocab_2)
        base_inv_grams_idxs = self._tokens_to_idxs(base_inv_grams, self.vocab.vocab_2)
        base_gap_grams_idxs = self._tokens_to_idxs(base_gap_grams, self.vocab.vocab_2)

        query.base_grams_idxs_list = (base_grams_idxs, base_inv_grams_idxs, base_gap_grams_idxs)

        # fill base_grams indices for synonims
        syn_grams, syn_inv_grams, syn_gap_grams = get_ngrams(query.synonim_tokens, ngram=2, inverted=True,
                                                             with_gap=True)
        syn_grams_idxs = self._tokens_to_idxs(syn_grams, self.vocab.vocab_2)
        syn_inv_grams_idxs = self._tokens_to_idxs(syn_inv_grams, self.vocab.vocab_2)
        syn_gap_grams_idxs = self._tokens_to_idxs(syn_gap_grams, self.vocab.vocab_2)

        query.syn_grams_idxs_list = (syn_grams_idxs, syn_inv_grams_idxs, syn_gap_grams_idxs)
        return query

    def predict_query(self, query: Query, doc_ids_pred: list):
        q_id = query.id
        pred = np.zeros(len(self.doc_ids_map))

        query = self.extend_query(query)
        #
        # # fill tokens indices
        # query.base_tokens_idxs = np.array([self.vocab.vocab_1[w] for w in query.tokens])
        # query.synonim_tokens_idxs = np.array([self.vocab.vocab_1[w] for w in query.synonim_tokens])
        #
        # # fill base_grams indices for base
        # base_grams, base_inv_grams, base_gap_grams = get_ngrams(query.tokens,ngram=2, inverted=True, with_gap=True)
        # base_grams_idxs = self._tokens_to_idxs(base_grams, self.vocab.vocab_2)
        # base_inv_grams_idxs = self._tokens_to_idxs(base_inv_grams, self.vocab.vocab_2)
        # base_gap_grams_idxs = self._tokens_to_idxs(base_gap_grams, self.vocab.vocab_2)
        #
        # query.base_grams_idxs_list = (base_grams_idxs,base_inv_grams_idxs, base_gap_grams_idxs)
        #
        # # fill base_grams indices for synonims
        # syn_grams, syn_inv_grams, syn_gap_grams = get_ngrams(query.synonim_tokens,ngram=2, inverted=True, with_gap=True)
        # syn_grams_idxs = self._tokens_to_idxs(syn_grams, self.vocab.vocab_2)
        # syn_inv_grams_idxs = self._tokens_to_idxs(syn_inv_grams, self.vocab.vocab_2)
        # syn_gap_grams_idxs = self._tokens_to_idxs(syn_gap_grams, self.vocab.vocab_2)
        #
        # query.syn_grams_idxs_list = (syn_grams_idxs,syn_inv_grams_idxs, syn_gap_grams_idxs)


        #for doc_id in self.doc_ids_map.dict.keys():
        for doc_id in doc_ids_pred:
            doc_index = self.doc_ids_map[doc_id]
            score = self.score_query_doc(query, doc_id)
            pred[doc_index] = score

        return q_id, pred

    def predict_multithread(self, queries:dict, processes = 4, top_n = 10):
        predicts = dict()

        pool = Pool(processes)
        tasks = list(queries.values())

        print('---')

        start_time = datetime.now()
        counter = 0
        total = len(tasks)

        for res in pool.imap(self.predict_query, tasks):
            counter += 1
            q_id, pred = res
            predicts[q_id] = pred
            elapsed_time = datetime.now() - start_time
            predict_time = (elapsed_time / (float(counter) )) * total

            print('counter: {0} , predict_time: {1} , elapsed_time: {2}'.format(counter,predict_time, elapsed_time))


        for q_id in predicts.keys():
            scores = predicts[q_id]
            top_doc_indexes = np.argsort(scores)[::-1][:top_n]
            top_doc_ids = [self.doc_ids_map.get_inverted(i) for i in top_doc_indexes]
            predicts[q_id] = top_doc_ids

        print('all_done!')

        return predicts

    def predict_onethread(self, queries:dict, sample_pred: dict,  top_n = 10):
        predicts = dict()

        tasks = list(queries.values())

        for query in tqdm(tasks):
            doc_ids_pred = sample_pred[query.id]
            res = self.predict_query(query, doc_ids_pred)
            q_id, pred = res
            predicts[q_id] = pred

        for q_id in queries.keys():
            scores = predicts[q_id]
            top_doc_indexes = np.argsort(scores)[::-1][:top_n]
            top_doc_ids = [self.doc_ids_map.get_inverted(i) for i in top_doc_indexes]
            predicts[q_id] = top_doc_ids

        print('all_done!')
        return predicts




    # def predict_query_matrix(self, query: Query):
    #     q_id = query.id
    #     pred = np.zeros(len(self.doc_ids_map))
    #
    #     query = self.extend_query(query)
    #
    #
    # def predict_matrix(self, queries: dict, top_n=10):
    #     scores_unigram = np.log(self.icf_unigram) * (self.tf_unigram / (self.tf_unigram + self.k1 + self.dl[:, None] / self.k2))
    #
    #     for q_id in queries.keys():
    #         pass

    # def predict(self, queries: dict, top_n = 10):
    #
    #     scores_unigram = np.log(self.icf_unigram) * (self.tf_unigram / (self.tf_unigram + self.k1 + self.dl[:, None] / self.k2))
    #     scores_unigram += np.log(self.icf_unigram) * (0.2 * self.hdr / (1.0 + self.hdr))
    #
    #     log_icf = np.log(self.icf_unigram)
    #     bin_tf = self.tf_unigram
    #     bin_tf[bin_tf > 0] = 1.0
    #
    #     scores_bigram = 0.3 * (np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[:, 0]])
    #                           + np.log(self.icf_unigram[self.indxs_bigrams_to_unigrams[:, 1]])) \
    #               * self.tf_bigram / (1.0 + self.tf_bigram)
    #
    #
    #
    #
    #
    #     pass
    #
    #
    #
    #
    # def predict(self, q_matrix_1, q_matrix_2, q_ids, top_n=10):
    #     pred = {}
    #     for i in range(q_matrix_1.shape[0]):
    #         q_row_1 = q_matrix_1[i]
    #         q_row_2 = q_matrix_2[i]
    #
    #         q_len = q_row_1[q_row_1 > 0].shape[0]
    #
    #         q_score_1 = np.matmul(self.scores_unigram, q_row_1)
    #         q_score_2 = np.matmul(self.scores2, q_row_2)
    #
    #         all_words_score = 0.2 * np.matmul(self.log_icf, q_row_1) * (
    #                     0.03 ** (q_len - np.matmul(self.bin_tf_unigram, q_row_1)))
    #
    #         q_score = q_score_1 + q_score_2 + all_words_score
    #
    #         top_scores_idxs = np.argsort(q_score)[::-1][:top_n]
    #         top_doc_ids = self.doc_ids[top_scores_idxs]
    #         pred[q_ids[i]] = top_doc_ids
    #     return pred






def _pred_query(args):
    model, query = args
    return model.predict_query(query)

from multiprocessing.dummy import Pool as ThreadPool


def model_predict_multithread(model, queries:dict, processes = 4, top_n = 10):
    predicts = dict()


    tasks = list(queries.values())
    tasks = list(zip([model]*len(tasks), tasks))

    print('---')

    start_time = datetime.now()
    counter = 0
    total = len(tasks)

    pool = ThreadPool(processes)
    for res in tqdm(pool.imap(_pred_query, tasks), total=len(tasks)):
        counter += 1
        q_id, pred = res
        predicts[q_id] = pred
        elapsed_time = datetime.now() - start_time
        predict_time = (elapsed_time / (float(counter) )) * total

        print('counter: {0} , predict_time: {1} , elapsed_time: {2}'.format(counter,predict_time, elapsed_time))


    for q_id in predicts.keys():
        scores = predicts[q_id]
        top_doc_indexes = np.argsort(scores)[::-1][:top_n]
        top_doc_ids = [model.doc_ids_map.get_inverted(i) for i in top_doc_indexes]
        predicts[q_id] = top_doc_ids

    print('all_done!')

    return predicts


PART_WEIGHTS = {TITLE: 3.0, TEXT: 1.0}
K1 = 1.0
K2 = 1000.0
B = 0.75




if __name__ == "__main__":
    print('started scorer!')
    data_folder = '../../../Data/hw1/'
    processed_folder = data_folder + 'new_data/text_documents/'
    fwd_index_folder = data_folder + 'new_data/statistics/fwd_index/'
    statistics_folder = data_folder + 'new_data/statistics/'

    predictions_folder = data_folder + 'new_data/predictions/'

    queries_filename = data_folder + 'queries.numerate_review.txt'

    sample_pred = load_predict('sample_sabmission.txt')

    queries, vocab = load_queries(queries_filename)

    docs_obj = pickle.load(open(data_folder + 'documents.pkl', 'rb'))  # get_documents(data_folder)
    documents = docs_obj.docs
    for d in documents:
        d.data_path = data_folder


    doc_ids_map = TwoWayDict(keys=list(docs_obj.docs_ids.keys()),
                             items=list(range(len(
                                 list(docs_obj.docs_ids.keys())
                             )))
                             )
    queries_ids_map = TwoWayDict(keys=list(queries.keys()),
                                 items=list(range(len(queries.keys())))
                                 )

    docs_num = len(doc_ids_map)
    queries_num = len(queries_ids_map)
    unigrams_num = len(vocab.vocab_1)


    counts_unigram = pickle.load(open(statistics_folder + 'unigram_counts.pkl', 'rb'))

    counts_bigram_raw = pickle.load(open(statistics_folder + 'bigram_counts_raw.pkl', 'rb'))
    counts_bigram_inv = pickle.load(open(statistics_folder + 'bigram_counts_inv.pkl', 'rb'))
    counts_bigram_gap = pickle.load(open(statistics_folder + 'bigram_counts_gap.pkl', 'rb'))

    document_lengths = pickle.load(open(statistics_folder + 'document_lengths.pkl', 'rb'))

    yandex_model = YandexModel(part_names=[TEXT, TITLE], part_weights=[1.0, 3.0],
                                vocab = vocab, k1 = 1.2, k2 = 1000.0, synonim_weight=0.7 )

    yandex_model.fit(counts_unigram, counts_bigram_raw, counts_bigram_inv,counts_bigram_gap,
                     document_lengths, doc_ids_map)


    print('prediction...')
    ya_predict = yandex_model.predict_onethread(queries, sample_pred, top_n=10)
    #ya_predict = model_predict_multithread(yandex_model, queries, processes=10, top_n=10)
    save_predict(ya_predict, predictions_folder + 'new_ya_prediction.txt')

    fen_pred = load_predict('../fen99_submission.txt')
    print(calc_ndcg(ya_predict, fen_pred).mean())

    # pred_q1 = yandex_model.predict_query(queries[1])
    #
    # print(pred_q1)


