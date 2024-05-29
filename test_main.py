import os
from dlnlp.app.TextSummary import TextSummary
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "dlnlp/datasets/sample.txt")
    ts = TextSummary(data_src=path)
    ts.load_data_set()
    ts.set_mlm_model("skl-td-idf")
    print(" ".join(ts.generate_summary()))
    print("---------------------------------")
    ts.set_mlm_model("torch-cnn")
    print(" ".join(ts.generate_summary()))
    print("---------------------------------")
    ts.set_mlm_model("torch-cnn-bert")
    print(" ".join(ts.generate_summary()))

