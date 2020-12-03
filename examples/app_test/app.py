from hatespans.app.hate_spans_app import HateSpansApp

app = HateSpansApp("small", use_cuda=False)
print(app.predict_hate_spans("This government is fucking crazy"))