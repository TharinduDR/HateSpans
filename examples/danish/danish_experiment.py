import pandas as pd

from hatespans.algo.evaluation import binary_macro_f1, binary_weighted_f1
from hatespans.app.hate_spans_app import HateSpansApp

test = pd.read_csv('data/offenseval-da-test-v1.tsv', sep="\t")
test = test.rename(columns={'tweet': 'text'})


test_sentences = test['text'].tolist()

app = HateSpansApp("multilingual-large", use_cuda=False)

hate_spans_list = []
predictions = []
for test_sentence in test_sentences:
    hate_spans = app.predict_hate_spans(test_sentence, spans=True, language="da")
    hate_spans_list.append(hate_spans)

    if len(hate_spans) > 0:
        predictions.append("OFF")

    else:
        predictions.append("NOT")


print("Macro F1 ", binary_macro_f1(test['subtask_a'].tolist(), predictions) )
print("Weighted F1 ", binary_weighted_f1(test['subtask_a'].tolist(), predictions) )


