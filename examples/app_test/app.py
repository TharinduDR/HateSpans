from hatespans.app.hate_spans_app import HateSpansApp

app = HateSpansApp("small", use_cuda=False)
print(app.predict_hate_spans("You motherfucking cunt. I will kill you", spans=True))