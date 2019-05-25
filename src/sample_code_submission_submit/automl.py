from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from deap import creator, tools
from CONSTANT import ENSEMBLE, ENSEMBLE_OBJ

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
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        # "is_unbalance": True
    }

    X_sample, y_sample = data_sample(X, y, 30000)

    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)

    if ENSEMBLE:
        hyperparams_li = hyperopt_lightgbm(X_sample, y_sample, params, config)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        config["model"] = [lgb.train({**params, **hyperparams},
                                     train_data,
                                     300,
                                     valid_data,
                                     early_stopping_rounds=30,
                                     verbose_eval=100,
                           init_model="%.10f" % (-hyperparams.fitness.values[0]))
                           for hyperparams in hyperparams_li]
    else:
        hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        config["model"] = lgb.train({**params, **hyperparams},
                                    train_data,
                                    500,
                                    valid_data,
                                    early_stopping_rounds=30,
                                    verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    if ENSEMBLE:
        return np.mean([model.predict(X) for model in config["model"]], axis=0)
    else:
        return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val)
    valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        # "is_unbalance": hp.choice("is_unbalance", [True, False]),
        # "scale_pos_weight": hp.loguniform('scale_pos_weight', np.log(np.sum(y == 0)/np.sum(y == 1)), 0)
        # if np.sum(y == 0)/(np.sum(y == 1) + 0.0001) > 1
        # else hp.loguniform('scale_pos_weight', 0, np.log(np.sum(y == 0)/np.sum(y == 1))),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300,
                          valid_data, early_stopping_rounds=30, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        result_dict = {'loss': -score, 'status': STATUS_OK}

        if ENSEMBLE:
            # save the model
            model.save_model("%.10f" % score, num_iteration=model.best_iteration)

            # predicts of valid set
            result_dict['predicts'] = np.round(model.predict(valid_data.data))

            if ENSEMBLE_OBJ == 3:
                # num of weak sub-models
                result_dict['num_trees'] = model.num_trees()

        return result_dict

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=10, verbose=1,
                         rstate=np.random.RandomState(1))

    if ENSEMBLE:
        # # select top half of the classifiers according to auc
        # trials._dynamic_trials.sort(key=lambda data: data['result']['loss'])
        # best_li = [trials._dynamic_trials[i]['misc']['vals'] for i in range(int(len(trials._dynamic_trials)/2))]
        # hyperparams_li = []
        # for best in best_li:
        #     for key in best:
        #         best[key] = best[key][0]
        #     hyperparams_li.append(space_eval(space, best))

        # select top half of the classifiers according to NCL
        predicts_ens = np.mean([trail['result']['predicts'] for trail in trials._dynamic_trials], axis=0)
        pop = []
        for trail in trials._dynamic_trials:
            hyperparams = trail['misc']['vals']
            for key in hyperparams:
                hyperparams[key] = hyperparams[key][0]
            hyperparams = space_eval(space, hyperparams)
            ind = creator.Individual(hyperparams)
            if ENSEMBLE_OBJ == 3:
                ind.fitness.values = (trail['result']['loss'],
                                      -np.sum((trail['result']['predicts'] - predicts_ens) ** 2),
                                      trail['result']['num_trees'])
            else:
                ind.fitness.values = (trail['result']['loss'],
                                      -np.sum((trail['result']['predicts'] - predicts_ens) ** 2))

            pop.append(ind)

        pop = tools.selNSGA2(pop, int(len(trials._dynamic_trials)/2))
        hyperparams_li = list(pop)

        return hyperparams_li
    else:
        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
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
