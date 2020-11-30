from spacy.lang.en import English


def predict_spans(model, text):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens = tokenizer(text)
    tokenised_text = []
    for token in tokens:
        tokenised_text.append(token.text)

    predictions, raw_outputs = model.predict(tokenised_text)
    print(predictions)