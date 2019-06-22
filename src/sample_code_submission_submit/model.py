import os

os.system("pip3 install --default-timeout=1000 scikit-learn==0.21.2")
os.system("pip3 install --default-timeout=1000 hyperopt")
os.system("pip3 install --default-timeout=1000 lightgbm")
os.system("pip3 install -U --default-timeout=1000 pandas==0.24.2")
os.system("pip3 install --default-timeout=1000 deap")
os.system("pip3 install --default-timeout=1000 category_encoders")

import copy
import numpy as np
import pandas as pd

from CONSTANT import MAIN_TABLE_NAME, \
    REDUCTION_SWITCH, \
    FEATURE_SELECTION_SWITCH, \
    DATA_BALANCE_SWITCH, \
    BAYESIAN_OPT,\
    ENSEMBLE, \
    ENSEMBLE_OBJ, \
    DATA_DOWNSAMPLING_SWITCH, TIME_PREFIX, MULTI_CAT_PREFIX, DROP_OUTLIER
from merge import merge_table
from preprocess import clean_df, \
    clean_tables, \
    feature_selection, \
    feature_engineer_rewrite, feature_engineer_rewrite_seq, \
    data_balance, \
    data_downsampling,\
    drop_outlier
from util import Config, log, show_dataframe, timeit
from deap import base, creator
import gc
gc.enable()

# if BAYESIAN_OPT:
#     from bayesml import predict, train, validate
# else:
#     from automl import predict, train, validate
from automl import predict, train, validate


class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        if ENSEMBLE:
            # for NSGA-II selection
            if ENSEMBLE_OBJ == 3:
                creator.create("FitnessMin", base.Fitness, weights=(-1, -1, -1))
            else:
                creator.create("FitnessMin", base.Fitness, weights=(-1, -1))
            creator.create("Individual", dict, fitness=creator.FitnessMin)
        self.pca = None
        self.scaler = None
        self.selected_features_0 = None
        self.selected_features_1 = None

    @timeit
    def fit(self, Xs, y, time_ramain):

        # self.tables = copy.deepcopy(Xs)
        self.tables = Xs
        if DATA_DOWNSAMPLING_SWITCH:
            self.tables[MAIN_TABLE_NAME], y = data_downsampling(self.tables[MAIN_TABLE_NAME], y, self.config)
        if DROP_OUTLIER:
            # the percentage of outliers dropped is around 15% to 20%
            inlier_lable = drop_outlier(clean_df(self.tables[MAIN_TABLE_NAME]))
            self.tables[MAIN_TABLE_NAME] = self.tables[MAIN_TABLE_NAME][inlier_lable == 1].reset_index(drop=True)
            y = y[inlier_lable == 1].reset_index(drop=True)
        clean_tables(self.tables)
        X = merge_table(self.tables, self.config)
        self.time_feature_list = [c for c in X if c.startswith(TIME_PREFIX)]
        self.mul_feature_list = [c for c in X if c.startswith(MULTI_CAT_PREFIX)]
        clean_df(X)

        if FEATURE_SELECTION_SWITCH:
            _, self.selected_features_0 = feature_selection(X.drop(columns=self.time_feature_list+self.mul_feature_list)
                                                            , y, self.config, 0.5)
        selected_features = list(self.selected_features_0) + self.time_feature_list + self.mul_feature_list

        X = feature_engineer_rewrite(X.filter(selected_features), self.config)

        if FEATURE_SELECTION_SWITCH:
            X, self.selected_features_1 = feature_selection(X, y , self.config, 0.7)

        train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table, len_X_train = Xs[MAIN_TABLE_NAME], len(Xs[MAIN_TABLE_NAME])
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'], sort=True)
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        # Xs[MAIN_TABLE_NAME] = clean_df(Xs[MAIN_TABLE_NAME])
        clean_df(Xs[MAIN_TABLE_NAME])
        X = merge_table(Xs, self.config)
        clean_df(X)
        selected_features = list(self.selected_features_0) + self.time_feature_list + self.mul_feature_list
        X = feature_engineer_rewrite(X.filter(selected_features), self.config)

        # X = X[X.index.str.startswith("test")]
        X = X.iloc[len_X_train:,]
        X.sort_index(inplace=True)
        if FEATURE_SELECTION_SWITCH:
            X = X[self.selected_features_1]
        result = predict(X, self.config)

        del self.tables, X_test
        # gc.collect()

        return pd.Series(result)
