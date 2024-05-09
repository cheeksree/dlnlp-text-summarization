from dlnlp.DataLoaders import get_content
from dlnlp.mlm.skl_tfidf import textSummary_m
class TextSummary:
    def __init__(self, data_src="", ml_module=""):
        self.data_src = data_src
        self.ml_module = ml_module
    def load_data_set(self):
        self.f_list = get_content(self.data_src)

    def load_mlm_module(self):
        if self.ml_module == "skl-td-idf":
            return textSummary_m(self.f_list)

