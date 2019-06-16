from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from CONSTANT import ENSEMBLE, ENSEMBLE_OBJ
from deap import creator, tools
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from CONSTANT import ENSEMBLE, \
    ENSEMBLE_OBJ, \
    train_lgb_params, \
    HYPEROPT_SEED, \
    DATA_SPLIT_SEED, SEED

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


def lgb_f1_score(preds, data):
    y_true = data.get_label()
    preds = np.round(preds)
    return 'f1', f1_score(y_true, preds), True

@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "binary",
        "metric": "auc",  # binary_logloss, auc
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 4,
        # "is_unbalance": True,
        # "scale_pos_weight": 2,
    }
    # params = train_lgb_params

    X_sample, y_sample = data_sample(X, y)

    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)

    if ENSEMBLE:
        hyperparams_li = hyperopt_lightgbm(X_sample, y_sample, params, config)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        config["model"] = [lgb.train({**params, **hyperparams},
                                     train_data,
                                     500,
                                     valid_data,
                                     early_stopping_rounds=20,
                                     verbose_eval=100,
                                     # feval=lgb_f1_score,
                                     # init_model=f"model_{hyperparams['ensemble_i']}"
                                     )
                           for hyperparams in hyperparams_li]
    else:
        hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        config["model"] = lgb.train({**params, **hyperparams},
                                    train_data,
                                    200,
                                    valid_data,
                                    early_stopping_rounds=20,
                                    verbose_eval=100,
                                    # feval=lgb_f1_score
                                    )


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    if ENSEMBLE:
        return np.mean([model.predict(X) for model in config["model"]], axis=0)
    else:
        return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    if ENSEMBLE:
        # global model_i
        # model_i = 0
        free_raw_data = False
    else:
        free_raw_data = True

    # X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.2)
    # train_data = lgb.Dataset(X, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=free_raw_data)

    # cross validation
    data = lgb.Dataset(X, label=y, free_raw_data=free_raw_data)
    data_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    train_data_li = []
    valid_date_li = []
    for train_indices, valid_indices in data_gen:
        train_data_li.append(data.subset(train_indices))
        valid_date_li.append(data.subset(valid_indices))
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.9)),
        "max_depth": hp.choice("max_depth", [1, 2, 3, 4, 5, 6, 7, 8]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 100, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 100, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        # "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1)
        # "colsample_bytree":
        # "min_data_in_leaf": hp.choice("min_data_in_leaf", np.linspace(5, 30, 5, dtype=int)),
        # "is_unbalance": hp.choice("is_unbalance", [True, False]),
        # "scale_pos_weight": hp.loguniform('scale_pos_weight', np.log(np.sum(y == 0)/np.sum(y == 1)), 0)
        # if np.sum(y == 0)/(np.sum(y == 1) + 0.0001) < 1
        # else hp.loguniform('scale_pos_weight', 0, np.log(np.sum(y == 0)/np.sum(y == 1))),
        # "scale_pos_weight": hp.quniform("feature_fraction", np.maximum(np.sum(y == 0)/np.sum(y == 1), 0.1), 1, 0.1)
        # if np.sum(y == 0)/(np.sum(y == 1) + 0.0001) < 1
        # else hp.quniform('scale_pos_weight', 1, np.sum(y == 0)/np.sum(y == 1), (np.sum(y == 0)/np.sum(y == 1) - 1)/10),
    }

    def objective(hyperparams):

        model_li = []
        for j in range(len(train_data_li)):
            model = lgb.train({**params, **hyperparams}, train_data_li[j], 300,
                              valid_date_li[j], early_stopping_rounds=20, verbose_eval=0,
                              # feval=lgb_f1_score
                              )
            model_li.append(model)

        score = np.mean([model.best_score["valid_0"][params["metric"]] for model in model_li])

        # in classification, less is better
        result_dict = {'loss': -score, 'status': STATUS_OK}

        if ENSEMBLE:
            # save the model
            # global model_i
            # model.save_model(f"model_{model_i}", num_iteration=model.best_iteration)
            # model_i = model_i + 1

            # predicts of valid set
            # result_dict['predicts'] = np.round(np.mean([model.predict(data.data) for model in model_li]))
            result_dict['predicts'] = np.mean([model.predict(data.data) for model in model_li], axis=0)

            if ENSEMBLE_OBJ == 3:
                # num of weak sub-models
                result_dict['num_trees'] = model.num_trees()

        return result_dict

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=10, verbose=-1,
                         rstate=np.random.RandomState(HYPEROPT_SEED))

    if ENSEMBLE:
        # # select top half of the classifiers according to auc
        # trials._dynamic_trials.sort(key=lambda data: data['result']['loss'])
        # best_li = [trial['misc']['vals']
        #            for trial in trials._dynamic_trials[0:int(len(trials._dynamic_trials)/2)]]
        # hyperparams_li = []
        # for best in best_li:
        #     for key in best:
        #         best[key] = best[key][0]
        #     hyperparams_li.append(space_eval(space, best))

        # select top half of the classifiers according to NCL
        predicts_ens = np.mean([trail['result']['predicts'] for trail in trials._dynamic_trials], axis=0)
        pop = []
        i = 0
        for trail in trials._dynamic_trials:
            hyperparams = trail['misc']['vals']
            for key in hyperparams:
                hyperparams[key] = hyperparams[key][0]
            hyperparams = space_eval(space, hyperparams)
            # hyperparams['ensemble_i'] = i
            ind = creator.Individual(hyperparams)
            # weights1 = (y_val == 1) * np.sum(y_val == 0) / len(y_val)
            # weights0 = (y_val == 0) * np.sum(y_val == 1) / len(y_val)
            # weights = weights0 + weights1
            if ENSEMBLE_OBJ == 3:
                ind.fitness.values = (trail['result']['loss'],
                                      -np.sum(((trail['result']['predicts'] - predicts_ens) ** 2)),
                                      trail['result']['num_trees'])
            else:
                ind.fitness.values = (trail['result']['loss'],
                                      -np.sum(((trail['result']['predicts'] - predicts_ens) ** 2)))

            pop.append(ind)
            i += 1
        pop = tools.selNSGA2(pop, int(len(trials._dynamic_trials)/2))
        hyperparams_li = list(pop)

        # # ensemble selection
        # num_classifiers = 0
        # sum_predicts = 0
        # hyperparams_li = []
        # dynamic_trials = trials._dynamic_trials.copy()
        # for i in range(len(dynamic_trials)):
        #     num_classifiers += 1
        #     best_trail = min(dynamic_trials,
        #                      key=lambda data:
        #                      np.mean(np.abs(valid_data.label -
        #                                     np.round((sum_predicts + data['result']['predicts']) /
        #                                              num_classifiers))))
        #     dynamic_trials = [trail for trail in dynamic_trials if trail != best_trail]
        #     sum_predicts += best_trail['result']['predicts']
        #     hyperparams = {}
        #     for key in best_trail['misc']['vals']:
        #         hyperparams[key] = best_trail['misc']['vals'][key][0]
        #     hyperparams = space_eval(space, hyperparams)
        #     hyperparams['ensemble_error'] = \
        #         np.mean(np.abs(valid_data.label - np.round(sum_predicts / num_classifiers)))
        #     hyperparams['ensemble_num'] = num_classifiers
        #     hyperparams_li.append(hyperparams)
        # ensemble_num = min(hyperparams_li, key=lambda data: data['ensemble_error'])['ensemble_num']
        # print(f"Ensemble_num: {ensemble_num}!!!")
        # hyperparams_li = hyperparams_li[0:ensemble_num]

        return hyperparams_li
    else:
        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=DATA_SPLIT_SEED)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=3000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=SEED)
        y_sample = y[X_sample.index]

        # # for unbalanced data - take care of imbalance
        # X_sample = X.assign(label=y)
        # rate = pd.DataFrame(data=[[1, len(y) - np.sum(y)], [0, np.sum(y)]],
        #                     columns=['label', 'rate'])
        # X_sample = X_sample.merge(rate, on='label') \
        #     .sample(nrows, random_state=1, weights='rate')
        # y_sample = X_sample['label']
        # X_sample = X_sample.drop(['label', 'rate'], axis=1)

        # # for unbalanced data - keep imbalance
        # nrows_1 = int(np.ceil(np.sum(y) / len(y) * np.sum(nrows)))
        # X_sample = X.assign(label=y)
        # X_sample_1 = X_sample \
        #     .query("label == 1") \
        #     .sample(nrows_1, random_state=1)
        # X_sample_0 = X_sample \
        #     .query("label == 0") \
        #     .sample(nrows - nrows_1, random_state=1)
        # X_sample = pd.concat([X_sample_0, X_sample_1],
        #                      sort=False,
        #                      ignore_index=True)
        # X_sample = shuffle(X_sample)
        # y_sample = X_sample['label']
        # X_sample = X_sample.drop(['label'], axis=1)
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
