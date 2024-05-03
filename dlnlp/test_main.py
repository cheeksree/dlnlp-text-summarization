import os
from dlnlp.text_summary.skl_tfidf import textSummary_m
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "datasets/sample.txt")
    print("".join(textSummary_m(path)))