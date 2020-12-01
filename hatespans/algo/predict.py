from spacy.lang.en import English


def predict_spans(model, text):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens = tokenizer(text)
    sentences = []
    tokenised_text = []
    for token in tokens:
        tokenised_text.append(token.text)
    sentences.append(tokenised_text)

    predictions, raw_outputs = model.predict(sentences)
    span_predictions = []

    for token in tokens:
        toxicness = predictions[token.text]
        if toxicness == "TOXIC":
            location = token.idx
            if len(span_predictions) > 0:
                last_index = span_predictions[-1]
                if location == last_index + 2:
                    span_predictions.append(location - 1)
            length = len(token.text)
            for i in range(length):
                span_predictions.append(location + i)
    return span_predictions
