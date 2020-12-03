[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

# HateSpans

We provide state-of-the-art models to detect toxic spans in text. We have evaluated our models on  Toxic Spanstask at SemEval 2021 (Task 5).

## Pretrained HateSpans Models

We will be keep releasing new models. Please keep in touch.

| Models   | Average F1    |
|----------|:-------------:|
| small    | 0.6652        |

## Prediction
Following code can be used to predict toxic spans in text. Upon executing, it will download the relevant model and return the toxic spans.   

```python
from hatespans.app.hate_spans_app import HateSpansApp

app = HateSpansApp("small", use_cuda=False)
print(app.predict_hate_spans("You motherfucking cunt", spans=True))
```


