import statistics

from sklearn.model_selection import train_test_split

from examples.english.transformer_configs import transformer_config, MODEL_TYPE, MODEL_NAME
from hatespans.algo.evaluation import f1
from hatespans.algo.hate_spans_model import HateSpansModel
from hatespans.algo.predict import predict_spans
from hatespans.algo.preprocess import read_datafile, format_data

train = read_datafile('examples/english/data/tsd_train.csv')
dev = read_datafile('examples//english/data/tsd_trial.csv')

train_df = format_data(train)
# train_df.to_csv("train_1.csv", sep='\t', encoding='utf-8', index=False)
tags = train_df['labels'].unique().tolist()

model = HateSpansModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=transformer_config)

if transformer_config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train_df, test_size=0.1,  shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train_df)

model = HateSpansModel(MODEL_TYPE, transformer_config["best_model_dir"], labels=tags, args=transformer_config)


scores = []
for n, (spans, text) in enumerate(dev):
    predictions = predict_spans(model, text)
    score = f1(predictions, spans)
    scores.append(score)


print('avg F1 %g' % statistics.mean(scores))







