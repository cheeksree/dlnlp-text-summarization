from dlnlp.dlnlp.DataLoaders import get_content
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import numpy as np

def convert_td_idf(f_list):
    tfd = tf(stop_words="english", strip_accents="ascii")
    return tfd.fit_transform(f_list).toarray()

def page_rank(sim_mat):
    d = 0.85
    min_diff = 1e-5
    steps = 200
    sim_mat = sim_mat/(sim_mat.sum(axis=1, keepdims=True))
    pr_vector = np.array([1] * len(sim_mat))
    prev_diff = 0
    for step in range(steps):
        pr_vector = (1-d) + d * np.matmul(sim_mat, pr_vector)
        if abs(prev_diff - sum(pr_vector)) < min_diff:
            break
        prev_diff = sum(pr_vector)
    return pr_vector

def textSummary_m(path_f):
    f_list = get_content(path_f)
    x = convert_td_idf(f_list)
    sim_mat = sklearn.metrics.pairwise.cosine_similarity(x)
    pr_vec = page_rank(sim_mat)
    w_dict = {}
    af_list =[]
    i = 0
    while(i<pr_vec.size):
        w_dict[pr_vec[i]] = i
        i = i + 1
    a_list = list(w_dict.keys())
    a_list.sort(reverse=True)
    i = 0
    len_t = len(a_list)
    if len_t > 10 :
        len_t = 10
    while(i<len_t):
        af_list.append(f_list[w_dict[a_list[i]]])
        i = i + 1
    return af_list
