import torch
import torchtext.vocab
from kmeans_pytorch import kmeans as km
import numpy as np
from torchtext.transforms import BERTTokenizer
from collections import OrderedDict

class customEmbed(torch.nn.Module):
    def __init__(self,sample,dim):
        super(customEmbed,self).__init__()
        self.embed = torch.nn.Embedding(sample,dim)

    def forward(self, x):
        out = self.embed(x)
        return out
def use_bert_tok(in_x):
    bert_tok = BERTTokenizer(vocab_path="https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt", \
                  do_lower_case=True, return_tokens=False)
    val = bert_tok([in_x])
    token_ids = []
    for itm in val[0]:
        token_ids.append(int(itm))
    return token_ids

def setup_cnn(voc_arr):
    num_wds_sent = 12
    cnv_1d = torch.nn.Conv1d(num_wds_sent, 1, 2, stride=1)
    num_wds = len(voc_arr)
    i = 0
    wds_tensor = torch.empty(1, 15)
    tt = torch.tensor(voc_arr, dtype=torch.long)
    em_size = torch.max(tt)
    embed_nn = customEmbed(em_size + 1, 16)
    in_tt_x1 = embed_nn(tt)
    while ((i < num_wds) and ((num_wds - i) > num_wds_sent)):
        in_tt_x2 = cnv_1d(in_tt_x1[i:i + num_wds_sent])
        wds_tensor = torch.vstack((wds_tensor, in_tt_x2))
        i = i + num_wds_sent
    conv_size = wds_tensor.size()[0]
    cnv_1d_2nd = torch.nn.Conv1d(conv_size, 1, 2, stride=1)
    wds_tensor = cnv_1d_2nd(wds_tensor)
    return wds_tensor

def load_tokenizer(f_list, type="custom"):
    wds_tensor = torch.empty(1, 14)
    if type == "custom":
        tok = torchtext.data.utils.get_tokenizer("basic_english")
        in_x = tok("".join(f_list))
        vocab = torchtext.vocab.build_vocab_from_iterator([in_x], specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        for line in f_list:
            in_x = tok(line)
            voc = vocab(in_x)
            if len(voc) >= 1:
                wds_tensor = torch.vstack((wds_tensor, setup_cnn(voc)))
    if type == "bert":
        for line in f_list:
            voc = use_bert_tok(line)
            if len(voc) >= 1:
                wds_tensor = torch.vstack((wds_tensor, setup_cnn(voc)))
    return wds_tensor
def generate_word_embeds(f_list, type="custom"):
    wds_tensor = load_tokenizer(f_list, type)
    af_list = []
    num_clusters = 10
    out_p = np.zeros((num_clusters,wds_tensor.shape[0]),dtype=np.int)
    id_x = km(wds_tensor[1:],num_clusters)
    c = 1
    j = 0
    for i in id_x[0]:
        out_p[i,j] = c
        c = c + 1
        j = j + 1

    a_list = np.sort(out_p.max(axis=1))
    for i in a_list:
        str = "".join(f_list[i - 1])
        af_list.append(str)
    return af_list