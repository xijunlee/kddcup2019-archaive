from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

from int_bayes import BayesianOptimizationIntKernel

from util import Config, log, timeit


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def BayesianSearch(evaluation, space, int_dim=None, num_iter=25, init_points=5):
    """Bayesian Optimizer"""
    # create a Bayesian optimization object, and define the scopes of hyperparameters
    bayes = BayesianOptimization(evaluation, space)
    # bayes = BayesianOptimizationIntKernel(evaluation, space, int_dim=int_dim)
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    hyperparams = bayes.max['params']

    return hyperparams


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }

    X_sample, y_sample = data_sample(X, y, 30000)
    hyperparams = bayes_lightgbm(X_sample, y_sample, params, config)
    # adjust some hyperparameters to integers
    hyperparams['max_depth'] = int(hyperparams['max_depth'])
    hyperparams['num_leaves'] = int(hyperparams['num_leaves'])
    hyperparams['bagging_freq'] = int(hyperparams['bagging_freq'])
    print(hyperparams)

    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                500,
                                valid_data,
                                early_stopping_rounds=30,
                                verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def bayes_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    space = {
        "learning_rate": (0.01, 0.5),
        "max_depth": (-1, 6),
        "num_leaves": (10, 200),
        "feature_fraction": (0.5, 1),
        "bagging_fraction": (0.5, 1),
        "bagging_freq": (0, 50),
        "reg_alpha": (0, 2),
        "reg_lambda": (0, 2),
        "min_child_weight": (0.5, 10),
    }

    int_dim = [1, 4, 6]

    def GBM_evaluate(learning_rate, max_depth, num_leaves, feature_fraction, bagging_fraction, bagging_freq, reg_alpha,
                     reg_lambda, min_child_weight):
        """definition of evaluation of lightGBM"""
        hyperparams = {}
        # hyerparameters generated from Bayesian optimizer
        hyperparams['learning_rate'] = float(learning_rate)
        hyperparams['max_depth'] = int(max_depth)
        hyperparams['num_leaves'] = int(num_leaves)
        hyperparams['feature_fraction'] = float(feature_fraction)
        hyperparams['bagging_fraction'] = float(bagging_fraction)
        hyperparams['bagging_freq'] = int(bagging_freq)
        hyperparams['reg_alpha'] = float(reg_alpha)
        hyperparams['reg_lambda'] = float(reg_lambda)
        hyperparams['min_child_weight'] = float(min_child_weight)

        # X_sample, y_sample = data_sample(X, y, 30000)
        X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=30, verbose_eval=0)
        # score = model.best_score["valid_0"][params["metric"]]
        preds = model.predict(X_val)
        score = roc_auc_score(y_val, preds)

        return score

    hyperparams = BayesianSearch(GBM_evaluate, space, int_dim=int_dim, num_iter=15, init_points=30)

    return hyperparams


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
