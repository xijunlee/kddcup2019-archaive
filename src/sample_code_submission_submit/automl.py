from typing import Dict, List
import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from deap import creator, tools

from CONSTANT import (ENSEMBLE,
                      ENSEMBLE_OBJ,
                      AUTO, STACKING,
                      STACKING_METHOD,
                      HPO_EVALS,
                      ENSEMBLE_SIZE,
                      STOCHASTIC_CV,
                      TRAIN_DATA_SIZE,
                      TIME_PREFIX,
                      TABLE_PREFIX,
                      DOUBLE_VAL,
                      SEED,
                      CATEGORY_PREFIX)
from preprocess import clean_df
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
        "num_threads": 4
        # "categorical_feature": [c for c in X if c.startswith(CATEGORY_PREFIX)]
        # "is_unbalance": True,
        # "scale_pos_weight": 2,
    }
    if ENSEMBLE:
        hyperparams_li = hyperopt_lightgbm(X, y, params, config)
        # hyperparams_li = smac_lightgbm(X, y, params, config)

        if STACKING:
            # X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
            #
            # train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            # valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

            # cross validation
            data = lgb.Dataset(X, label=y, free_raw_data=False, categorical_feature=[c for c in X if c.startswith(CATEGORY_PREFIX)])
            data_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
            train_data_li = []
            valid_date_li = []
            valid_indices_li = []
            for train_indices, valid_indices in data_gen:
                train_data_li.append(data.subset(train_indices))
                valid_date_li.append(data.subset(valid_indices))
                valid_indices_li.append(valid_indices)
            data_indices = np.concatenate(valid_indices_li)

            config["model"] = [[lgb.train({**params, **hyperparams, **{"learning_rate": 0.1, "num_boost_round": 300}},
                                         train_data_li[i],
                                         300,
                                         valid_date_li[i],
                                         early_stopping_rounds=30,
                                         verbose_eval=100,
                                         callbacks=[lgb.reset_parameter(learning_rate=learning_rate_decay)],
                                         # feval=lgb_f1_score,
                                         # init_model=f"model_{hyperparams['ensemble_i']}"
                                         ) for i in range(5)]
                               for hyperparams in hyperparams_li]

            predicts = np.zeros(len(y))
            predicts_li = []
            for model in config["model"]:
                predicts[data_indices] = np.concatenate([model[i].predict(valid_date_li[i].data) for i in range(5)])
                predicts_li.append(predicts)
            ys = np.transpose(np.array(predicts_li))
            if STACKING_METHOD == 0:
                class_weight = {1: 1 - np.sum(y) / len(y), 0: np.sum(y) / len(y)}
                config["stacker_model"] = LogisticRegression(class_weight=class_weight, n_jobs=4,
                                                             max_iter=500, random_state=SEED).fit(ys, y)
            else:
                ys_train, ys_val, y_train, y_val = data_split(ys, y, 0.1)
                train_data = lgb.Dataset(ys_train, label=y_train, free_raw_data=False)
                valid_data = lgb.Dataset(ys_val, label=y_val, free_raw_data=False)
                config["stacker_model"] = lgb.train({**params, **{"learning_rate": 0.1, "num_boost_round": 300}},
                                         train_data,
                                         300,
                                         valid_data,
                                         early_stopping_rounds=30,
                                         verbose_eval=100,
                                         callbacks=[lgb.reset_parameter(learning_rate=learning_rate_decay)],
                                         # feval=lgb_f1_score,
                                         # init_model=f"model_{hyperparams['ensemble_i']}"
                                         )
        else:
            X_train, X_val, y_train, y_val = data_split(X, y, 0.2)

            # hyperparams_li = smac_lightgbm(X_sample, y_sample, params, config)
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

            config["model"] = [lgb.train({**params, **hyperparams, **{"learning_rate": 0.1, "num_boost_round": 300}},
                                         train_data,
                                         300,
                                         valid_data,
                                         early_stopping_rounds=30,
                                         verbose_eval=100,
                                         callbacks=[lgb.reset_parameter(learning_rate=learning_rate_decay)],
                                         # feval=lgb_f1_score,
                                         # init_model=f"model_{hyperparams['ensemble_i']}"
                                         )
                               for hyperparams in hyperparams_li]
    else:
        X_train, X_val, y_train, y_val = data_split(X, y, 0.2)

        hyperparams = hyperopt_lightgbm(X, y, params, config)
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=True)

        config["model"] = lgb.train({**params, **hyperparams, **{"learning_rate": 0.1, "num_boost_round": 500}},
                                    train_data,
                                    500,
                                    valid_data,
                                    early_stopping_rounds=30,
                                    verbose_eval=100,
                                    callbacks=[lgb.reset_parameter(learning_rate=learning_rate_decay)],
                                    # feval=lgb_f1_score
                                    )


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    if ENSEMBLE:
        if STACKING:
            Xs = np.transpose(np.array([np.mean([model[i].predict(X) for i in range(5)], axis=0) for model in config["model"]]))
            if STACKING_METHOD == 0:
                predicts = config["stacker_model"].predict_proba(Xs)
                ys = np.zeros(len(Xs))
                ys[predicts[:, 0] >= 0.5] = config["stacker_model"].classes_[0]
                ys[predicts[:, 1] >= 0.5] = config["stacker_model"].classes_[1]
                return ys
            else:
                return config["stacker_model"].predict(Xs)
        else:
            return np.mean([model.predict(X) for model in config["model"]], axis=0)
    else:
        return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    free_raw_data = False if ENSEMBLE else True

    if DOUBLE_VAL:
        if len(X) > 10 * TRAIN_DATA_SIZE:
            X, X_val_double, y, y_val_double = data_split(X, y, test_size=TRAIN_DATA_SIZE, random_state=SEED)
        else:
            X, X_val_double, y, y_val_double = data_split(X, y, test_size=0.1, random_state=SEED)
    X_, y_ = data_sample(X, y, TRAIN_DATA_SIZE, random_state=SEED)
    data = lgb.Dataset(X_, label=y_, free_raw_data=free_raw_data)

    # cross validation
    if STOCHASTIC_CV:
        data_all = lgb.Dataset(X, label=y, free_raw_data=free_raw_data)

    else:
        # if DOUBLE_VAL:
        #     X, X_val_double, y, y_val_double = data_split(X, y, 0.1)
        # data = lgb.Dataset(X, label=y, free_raw_data=free_raw_data)
        # if DOUBLE_VAL:
        #     data_val_double = lgb.Dataset(X_val_double, label=y_val_double, free_raw_data=free_raw_data)
        data_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)

        train_data_li = []
        valid_data_li = []
        valid_indices_li = []
        for train_indices, valid_indices in data_gen:
            train_data_li.append(data.subset(train_indices))
            valid_data_li.append(data.subset(valid_indices))
            valid_indices_li.append(valid_indices)
        data_indices = np.concatenate(valid_indices_li)

    def objective(hyperparams):
        model_li = []
        if STOCHASTIC_CV:
            X_train, X_val, y_train, y_val = data_split(X_, y_, test_size=0.2, random_state=SEED)
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
            valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=True)
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0,
                              # feval=lgb_f1_score
                              )
            model_li.append(model)
        else:
            for j in range(len(train_data_li)):
                model = lgb.train({**params, **hyperparams}, train_data_li[j], 300,
                                  valid_data_li[j], early_stopping_rounds=30, verbose_eval=0,
                                  # feval=lgb_f1_score
                                  )
                model_li.append(model)

        score = np.mean([model.best_score["valid_0"][params["metric"]] for model in model_li])

        # in classification, less is better
        result_dict = {'loss': -score, 'status': STATUS_OK}

        if DOUBLE_VAL:
            result_dict['double_val_loss'] = np.mean([roc_auc_score(y_val_double, model.predict(X_val_double))
                                                      for model in model_li])

        if ENSEMBLE:
            # save the model
            # global model_i
            # model.save_model(f"model_{model_i}", num_iteration=model.best_iteration)
            # model_i = model_i + 1

            # predicts of valid set
            # result_dict['predicts'] = np.mean([model.predict(data.data) for model in model_li], axis=0)
            if STOCHASTIC_CV:
                if len(model_li) == 1:
                    result_dict['predicts'] = model_li[0].predict(data_all.data)
                else:
                    result_dict['predicts'] = np.mean(model_li[0].predict(data_all.data), axis=0)
            else:
                result_dict['predicts'] = np.zeros(data.num_data())
                result_dict['predicts'][data_indices] = np.concatenate([model_li[j].predict(valid_data_li[j].data)
                                                                        for j in range(len(valid_data_li))])

            if ENSEMBLE_OBJ == 3:
                # num of weak sub-models
                result_dict['num_trees'] = model.num_trees()

        return result_dict

    if AUTO:
        params.update({
            "max_depth": 6,
            "num_leaves": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "learning_rate": 0.1
        })
        cv_results = lgb.cv({**params}, data, 500, nfold=5, metrics="auc", early_stopping_rounds=30, verbose_eval=0)
        params["num_boost_round"] = len(cv_results["auc-mean"])
        print("best_cv_score: ", cv_results["auc-mean"][-1])

        trial_li = []

        # space = {
        #     "scale_pos_weight": hp.uniform("scale_pos_weight", np.sum(y == 0) / np.sum(y == 1) * 2, 1)
        #     if np.sum(y == 0) / (np.sum(y == 1) + 0.0001) < 1
        #     else hp.uniform('scale_pos_weight', 1, np.sum(y == 0) / np.sum(y == 1) / 2),
        # }
        # trials = Trials()
        # best = hyperopt.fmin(fn=objective, space=space, trials=trials,
        #                      algo=hyperopt.tpe.suggest, max_evals=6, verbose=1,
        #                      rstate=np.random.RandomState(1))
        # for trial in trials._dynamic_trials:
        #     trial_li.append({'result': trial['result'],
        #                      'hyperparams': {**params,
        #                                      **space_eval(space,
        #                                                   {key: value[0] for key, value
        #                                                    in trial['misc']['vals'].items()})}})
        # params.update(space_eval(space, best))
        # print("beat_params: ", params)

        space = {
            "max_depth": hp.choice("max_depth", [2, 4, 6, 8]),
            "num_leaves": hp.choice("num_leaves", range(5, 200, 20)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.9, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 0.9, 0.1),
            "bagging_freq": hp.choice("bagging_freq", [1, 2, 4, 7, 10]),
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "boosting_type": hp.choice("boosting_type", ["gbdt", "rf"]),
            # "num_boost_round": hp.choice("num_boost_round", range(50, 500, 50)),
            # "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            # "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            # "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            # "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            # "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            # "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            # "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            # "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            # "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
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

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=hyperopt.tpe.suggest, max_evals=HPO_EVALS, verbose=1,
                             rstate=np.random.RandomState(1))
        for trial in trials._dynamic_trials:
            trial_li.append({'result': trial['result'],
                             'hyperparams': {**params,
                                             **space_eval(space,
                                                          {key: value[0] for key, value
                                                           in trial['misc']['vals'].items()})}})
        params.update(space_eval(space, best))
        print("best_params: ", params)
    else:
        params.update({
            "max_depth": 6,
            "num_leaves": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "learning_rate": 0.1
        })
        cv_results = lgb.cv({**params}, data, 500, nfold=5, metrics="auc", early_stopping_rounds=30, verbose_eval=0)
        params["num_boost_round"] = len(cv_results["auc-mean"])
        print("beat_cv_score: ", cv_results["auc-mean"][-1])

        trial_li = []

        space = {
            "max_depth": hp.choice("max_depth", [3, 5, 7, 9]),
            "num_leaves": hp.choice("num_leaves", range(5, 200, 20)),
        }
        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=hyperopt.tpe.suggest, max_evals=6, verbose=1,
                             rstate=np.random.RandomState(1))
        for trial in trials._dynamic_trials:
            trial_li.append({'result': trial['result'],
                             'hyperparams': {**params,
                                             **space_eval(space,
                                                          {key: value[0] for key, value
                                                           in trial['misc']['vals'].items()})}})
        params.update(space_eval(space, best))
        print("beat_params: ", params)

        space = {
            "min_child_weight": hp.loguniform("min_child_weight", np.log(0.001), np.log(1)),
            "min_data_in_leaf": hp.choice('min_data_in_leaf', range(10, 50, 4))
        }
        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=hyperopt.tpe.suggest, max_evals=6, verbose=1,
                             rstate=np.random.RandomState(1))
        for trial in trials._dynamic_trials:
            trial_li.append({'result': trial['result'],
                             'hyperparams': {**params,
                                             **space_eval(space,
                                                          {key: value[0] for key, value in
                                                           trial['misc']['vals'].items()})}})
        params.update(space_eval(space, best))
        print("beat_params: ", params)

        space = {
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.9, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", [0, 1, 2, 4, 7, 10]),
        }
        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=hyperopt.tpe.suggest, max_evals=6, verbose=1,
                             rstate=np.random.RandomState(1))
        for trial in trials._dynamic_trials:
            trial_li.append({'result': trial['result'],
                             'hyperparams': {**params,
                                             **space_eval(space,
                                                          {key: value[0] for key, value in
                                                           trial['misc']['vals'].items()})}})
        params.update(space_eval(space, best))
        print("beat_params: ", params)

        space = {
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        }
        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=hyperopt.tpe.suggest, max_evals=6, verbose=1,
                             rstate=np.random.RandomState(1))
        for trial in trials._dynamic_trials:
            trial_li.append({'result': trial['result'],
                             'hyperparams': {**params,
                                             **space_eval(space,
                                                          {key: value[0] for key, value in
                                                           trial['misc']['vals'].items()})}})
        params.update(space_eval(space, best))
        print("beat_params: ", params)

    if ENSEMBLE:
        # # Method1: select top half of the classifiers according to auc
        # trials._dynamic_trials.sort(key=lambda data: data['result']['loss'])
        # best_li = [trial['misc']['vals']
        #            for trial in trials._dynamic_trials[0:ENSEMBLE_SIZE]]
        # hyperparams_li = []
        # for best in best_li:
        #     for key in best:
        #         best[key] = best[key][0]
        #     hyperparams_li.append(space_eval(space, best))

        # Method2: select top half of the classifiers according to NCL
        predicts_ens = np.mean([trail['result']['predicts'] for trail in trial_li], axis=0)
        pop = []
        i = 0
        for trail in trial_li:
            hyperparams = trail['hyperparams']
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
        pop = tools.selNSGA2(pop, ENSEMBLE_SIZE)
        # pop = tools.selNSGA2(pop, 20)
        hyperparams_li = list(pop)

        # # Method3: ensemble selection
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
        if DOUBLE_VAL:
            min_double_val_loss = 0
            hyperparams = None
            for trial in trial_li:
                if trial['result']['double_val_loss'] < min_double_val_loss:
                    hyperparams = trial['hyperparams']
                    min_double_val_loss = trial['result']['double_val_loss']
        else:
            hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams



def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, random_state=SEED):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int = 5000, method: int = 2, random_state=SEED):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        if method == 0:
            X_sample = X.sample(nrows, random_state=random_state)
            y_sample = y[X_sample.index]
        elif method == 1:
            # for unbalanced data - take care of imbalance
            X_sample = X.assign(label=y)
            rate = pd.DataFrame(data=[[1, len(y) - np.sum(y)], [0, np.sum(y)]],
                                columns=['label', 'rate'])
            X_sample = X_sample.merge(rate, on='label') \
                .sample(nrows, random_state=random_state, weights='rate')
            y_sample = X_sample['label']
            X_sample = X_sample.drop(['label', 'rate'], axis=1)
        else:
            # for unbalanced data - keep imbalance
            nrows_1 = int(np.ceil(np.sum(y) / len(y) * np.sum(nrows)))
            X_sample = X.assign(label=y)
            X_sample_1 = X_sample \
                .query("label == 1") \
                .sample(nrows_1, random_state=random_state)
            X_sample_0 = X_sample \
                .query("label == 0") \
                .sample(nrows - nrows_1, random_state=random_state)
            X_sample = pd.concat([X_sample_0, X_sample_1],
                                 sort=False,
                                 ignore_index=True)
            X_sample = shuffle(X_sample)
            y_sample = X_sample['label']
            X_sample = X_sample.drop(['label'], axis=1)
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


def learning_rate_decay(current_itr):
    starting_rate = 0.1
    current_rate = starting_rate * (0.995 ** current_itr)
    return current_rate if current_rate > 1e-2 else 1e-2
