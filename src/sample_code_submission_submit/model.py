import os

os.system("apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC")
os.system("apt-get --assume-yes update")
os.system("apt-get --assume-yes install apt-utils libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev libdpkg-perl libgcc_s.so.1")
os.system("apt-get --assume-yes autoremove")

os.system("pip3 install -U --default-timeout=1000 pandas")
os.system("pip3 install --default-timeout=1000 deap")
os.system("pip3 install --default-timeout=1000 hyperopt")
os.system("pip3 install --default-timeout=1000 lightgbm")
os.system("pip3 install --default-timeout=1000 shap")

import copy
import numpy as np
import pandas as pd

from CONSTANT import MAIN_TABLE_NAME, \
    REDUCTION_SWITCH, \
    FEATURE_SELECTION_SWITCH, \
    DATA_BALANCE_SWITCH, \
    BAYESIAN_OPT,\
    ENSEMBLE, \
    ENSEMBLE_OBJ
from merge import merge_table
from preprocess import clean_df, \
    clean_tables, \
    feature_selection, \
    feature_engineer_rewrite
from util import Config, log, show_dataframe, timeit
from deap import base, creator

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
        self.selected_features = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        # self.tables = copy.deepcopy(Xs)
        self.tables = Xs
        main_table = Xs[MAIN_TABLE_NAME]

        # if DATA_BALANCE_SWITCH:
        #     main_table, y = data_balance(main_table, y, self.config)
        #     Xs[MAIN_TABLE_NAME] = main_table
        # there is no need doing data balance here
        clean_tables(self.tables)
        X = merge_table(self.tables, self.config)
        # X = clean_df(X)

        # if FEATURE_ENGINEERING_FT_SWITCH:
        #     X = feature_engineer_ft(X, self.config)
        # else:
        #     X = feature_engineer_base(X, self.config)

        X = feature_engineer_rewrite(X, self.config)

        if FEATURE_SELECTION_SWITCH:
            X, self.selected_features = feature_selection(X, y, self.config)
        if REDUCTION_SWITCH:
            X, self.scaler, self.pca = data_reduction_train(X)
        train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table, len_X_train = Xs[MAIN_TABLE_NAME], len(Xs[MAIN_TABLE_NAME])
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'], sort=True)
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        # clean_tables({"main_table":Xs[MAIN_TABLE_NAME]})
        Xs[MAIN_TABLE_NAME] = clean_df(Xs[MAIN_TABLE_NAME])
        X = merge_table(Xs, self.config)

        # X = clean_df(X)
        # if FEATURE_ENGINEERING_FT_SWITCH:
        #     X = feature_engineer_ft(X, self.config)
        # else:
        #     X = feature_engineer_base(X, self.config)

        X = feature_engineer_rewrite(X, self.config)

        # X = X[X.index.str.startswith("test")]
        X = X.iloc[len_X_train:,]
        X.sort_index(inplace=True)
        if FEATURE_SELECTION_SWITCH:
            X = X[self.selected_features]
        result = predict(X, self.config)

        return pd.Series(result)
