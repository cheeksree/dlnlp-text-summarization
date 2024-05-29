from ..DataLoaders import get_content
from ..mlm.skl_tfidf import textSummary_m
from ..mlm.torch_cnn import generate_word_embeds
class TextSummary:
    def __init__(self, data_src="", ml_module="",
                 data_src_type="txt"):
        self.data_src = data_src
        self.ml_module = ml_module
        self.data_src_type = data_src_type
    def load_data_set(self):
        if self.data_src_type == "txt":
            self.f_list = get_content(self.data_src)

    def set_mlm_model(self,model_str):
        self.ml_module=model_str

    def generate_summary(self):
        if self.ml_module == "skl-td-idf":
            return textSummary_m(self.f_list)
        if self.ml_module == "torch-cnn":
            return generate_word_embeds(self.f_list)
        if self.ml_module == "torch-cnn-bert":
            return generate_word_embeds(self.f_list, type="bert")


