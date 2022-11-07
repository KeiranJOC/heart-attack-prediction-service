import pickle
import bentoml
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


logging.basicConfig(level='INFO')

DATA_PATH = './data/heart.csv'
MODEL_PATH = './models/xgb.bin'

xgb_parameters = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}


def prepare_features(data_path):
    df = pd.read_csv(data_path)

    categorical = [
        'sex',
        'exng',
        'cp',
        'fbs',
        'restecg',
        'thall',
    ]

    numerical = [
        'age',
        'caa',
        'trtbps',
        'chol',
        'thalachh',
        'oldpeak',
        'slp',
    ]

    target = 'output'

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    
    y_train = df_train[target].values
    y_test = df_test[target].values

    del df_train['output']
    del df_test['output']

    train_dicts = df_train.to_dict(orient='records')
    test_dicts = df_test.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    return X_train, y_train, X_test, y_test, dv


def train(X_train, y_train, parameters):
    d_train = xgb.DMatrix(X_train, label=y_train)

    model = xgb.train(
        parameters,
        d_train, 
        num_boost_round=30,
        verbose_eval=5,
    )

    return model


def predict(X_test, y_test, model):
    d_test = xgb.DMatrix(X_test, label=y_test)

    y_pred = model.predict(d_test)

    return y_pred


def save_model(model, dv, model_path):
    with open(model_path, 'wb') as f_out:
        pickle.dump((model, dv), f_out)


if __name__=='__main__':
    logging.info('Preparing features')
    X_train, y_train, X_test, y_test, dv = prepare_features(DATA_PATH)

    logging.info('Training model')
    model= train(X_train, y_train, xgb_parameters)

    logging.info('Predicting on test set')
    y_pred = predict(X_test, y_test, model)
    auc = roc_auc_score(y_test, y_pred)
    logging.info(f'AUC: {auc}')

    # logging.info(f'Saving model to {MODEL_PATH}')
    # save_model(model, dv, MODEL_PATH)

    logging.info('Saving model with BentoML')
    bentoml.xgboost.save_model(
        'heart_attack_prediction_model', 
        model,
        custom_objects={
            'dictVectorizer': dv
        }
    )
