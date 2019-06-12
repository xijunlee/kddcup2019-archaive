import os
# import subprocess
# mem_available = subprocess.run('cat /proc/meminfo | grep MemAvailable', shell=True).stdout

str_sh1 = "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC"
str_sh2= "apt-get --assume-yes update"
str_sh3 = "apt-get --assume-yes install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip swig"

os.system(str_sh1)
os.system(str_sh2)
os.system(str_sh3)

# os.system("apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC")
# os.system("apt-get --assume-yes update")
# os.system("apt-get --assume-yes install swig libdpkg-perl")

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install deap")
os.system("pip3 install eli5")
os.system("pip3 install sklearn")
os.system("pip3 install category_encoders")
os.system("pip3 install bayesian-optimization")
# os.system("pip3 install smac")
# os.system("pip3 install numbers")
os.system("pip3 install psutil")
os.system("pip3 install featuretools")


import copy
import numpy as np
import pandas as pd

from CONSTANT import MAIN_TABLE_NAME, \
    REDUCTION_SWITCH, \
    FEATURE_GENERATION_SWITCH, \
    FEATURE_SELECTION_SWITCH, \
    DATA_BALANCE_SWITCH, \
    BAYESIAN_OPT,\
    DATA_DOWNSAMPLING_SWITCH, \
    ENSEMBLE, \
    ENSEMBLE_OBJ, \
    FEATURE_ENGINEERING_BASE_SWITCH, FEATURE_ENGINEERING_FT_SWITCH
from merge import merge_table
from preprocess import clean_df, \
    clean_tables, \
    feature_engineer_base, \
    data_reduction_train, \
    data_reduction_test, \
    feature_generation, \
    feature_selection, \
    data_balance, \
    data_downsampling, \
    feature_engineer_ft
from util import Config, log, show_dataframe, timeit
from deap import base, creator
import featuretools as ft

if BAYESIAN_OPT:
    from bayesml import predict, train, validate
else:
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
        self.tables = copy.deepcopy(Xs)
        main_table = Xs[MAIN_TABLE_NAME]

        # if DATA_BALANCE_SWITCH:
        #     main_table, y = data_balance(main_table, y, self.config)
        #     Xs[MAIN_TABLE_NAME] = main_table
        # there is no need doing data balance here
        # feature selection does the data balance and data downsampling
        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        if FEATURE_ENGINEERING_FT_SWITCH:
            X = feature_engineer_ft(X, self.config)
        else:
            X = feature_engineer_base(X, self.config)
        # if FEATURE_GENERATION_SWITCH:
        #     X, self.random_features = feature_generation(X)
        if FEATURE_SELECTION_SWITCH:
            X, self.selected_features = feature_selection(X, y, self.config)
        if REDUCTION_SWITCH:
            X, self.scaler, self.pca = data_reduction_train(X)
        train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table, len_X_train = Xs[MAIN_TABLE_NAME], len(Xs[MAIN_TABLE_NAME])
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)
        if FEATURE_ENGINEERING_FT_SWITCH:
            X = feature_engineer_ft(X, self.config)
        else:
            X = feature_engineer_base(X, self.config)
        # X = X[X.index.str.startswith("test")]
        X = X.iloc[len_X_train:,]
        # X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        # if FEATURE_GENERATION_SWITCH:
        #     X = feature_generation(X, self.random_features)
        if FEATURE_SELECTION_SWITCH:
            X = X[self.selected_features]
        if REDUCTION_SWITCH:
            X = data_reduction_test(X, self.scaler, self.pca)
        result = predict(X, self.config)

        return pd.Series(result)
