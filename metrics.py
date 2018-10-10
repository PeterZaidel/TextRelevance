import numpy as np

def load_predict(filename):
    pred = {}
    file = open(filename, 'r')
    for l in file.readlines()[1:]:
        qid, did = l.replace('\n', '').split(',')
        qid = int(qid)
        did = int(did)
        p = pred.get(qid, [])
        p.append(did)
        pred[qid] = p
    file.close()
    return pred

def save_predict(pred, filename):
    file = open(filename, 'w')
    file.write('QueryId,DocumentId\n')
    for qid in pred.keys():
        doc_ids = pred[qid]
        for did in doc_ids:
            file.write('{0},{1}\n'.format(qid, did))
    file.close()

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def calc_ndcg(my_pred, fen_pred, k=10):
    scores = []
    for qid in my_pred.keys():
        my_docs = my_pred[qid][:k]
        fen_docs = fen_pred[qid][:k]
        bin_rel = np.zeros(len(my_docs))
        for i in range(len(my_docs)):
            if my_docs[i] == fen_docs[i]:
                bin_rel[i] = 1.0

        score = ndcg_at_k(bin_rel, k)
        scores.append(score)
    return np.array(scores)


#
# def _gen_icf_unigram(unigram_counts, docs_num, tokens_num):
#     sum_counts = np.zeros((docs_num, tokens_num))
#     for pname in [TEXT]:
#         sum_counts += np.array(unigram_counts[pname])
#
#     icf = sum_counts.sum(0)
#     total_count = icf.sum()
#     icf[icf == 0] = 1.0
#     icf = total_count/icf
#
#     return icf
#
# def _gen_doc_lens( doc_lens, docs_num):
#     dl = np.zeros(docs_num)
#     for pn in [TEXT, TITLE]:
#         dl += doc_lens[pn]
#
#     return dl
#
# # def _gen_icf_bigram(icf_unigram, pairs_to_idxs):
# #     # icf_bigram = np.zeros((icf_unigram.shape[0], pairs_to_idxs.shape[0]))
# #     icf_bigram = icf_unigram[pairs_to_idxs[:, 0]] * icf_unigram[pairs_to_idxs[:, 1]]
# #     return icf_bigram
#
# def _gen_indexes_bigram_to_unigrams(vocab: Vocab):
#     bigram_to_unigrams = np.empty((len(vocab.vocab_2), 2), dtype=int)
#     for w2 in vocab.vocab_2:
#         w2_idx = vocab.vocab_2.dict[w2]
#         w21 = w2[0]
#         w22 = w2[1]
#
#         w21_idx = vocab.vocab_1.dict[w21]
#         w22_idx = vocab.vocab_1.dict[w22]
#
#         bigram_to_unigrams[w2_idx] = np.array([w21_idx, w22_idx])
#     return  bigram_to_unigrams
#
# def _gen_tf_unigram(unigram_counts):
#     sum_counts = np.zeros_like(unigram_counts[TEXT])
#     for pname in [TEXT]:
#         sum_counts += np.array(unigram_counts[pname])
#
#     return sum_counts
#
# def _gen_tf_bigrams(bigram_counts):
#     sum_counts = np.zeros_like(bigram_counts[TEXT])
#     for pname in [TEXT]:
#         sum_counts += np.array(bigram_counts[pname])
#
#     return sum_counts
#
# def _gen_hdr(part_coeffs_dict, unigram_counts):
#     hdr = np.zeros_like(unigram_counts[TITLE].toarray())
#     for pname in [TITLE]:
#         hdr += part_coeffs_dict[pname] * np.array(unigram_counts[pname].toarray())
#
#     return hdr
#
#
# class ModelData:
#     queries_ids_map = None
#     doc_ids_map = None
#
#     docs_num = None
#     queries_num = None
#     unigrams_num = None
#
#     vocab = None
#
#     tf_bigrams = None
#     tf_unigrams = None
#     icf_unigram = None
#     hdr = None
#
#     bigram_to_unigrams = None