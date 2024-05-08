import os
from dlnlp.dlnlp.app.text_summary.TextSummary import TextSummary
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "datasets/sample.txt")
    ts = TextSummary(data_src=path, ml_module="skl-td-idf")
    ts.load_data_set()
    print("".join(ts.load_mlm_module()))