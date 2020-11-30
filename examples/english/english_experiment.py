from sklearn.model_selection import train_test_split

from examples.english.transformer_configs import transformer_config, MODEL_TYPE, MODEL_NAME
from hatespans.algo.hate_spans_model import HateSpansModel
from hatespans.algo.preprocess import read_datafile, format_data

train = read_datafile('examples/english/data/tsd_train.csv')
dev = read_datafile('examples/english/data/tsd_trial.csv')

train_df = format_data(train)
# train_df.to_csv("train_1.csv", sep='\t', encoding='utf-8', index=False)
tags = train_df['labels'].unique()

model = HateSpansModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=transformer_config)

if transformer_config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train, test_size=0.1,  shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train_df)



