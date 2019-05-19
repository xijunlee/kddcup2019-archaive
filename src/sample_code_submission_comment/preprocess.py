import datetime

import CONSTANT
from util import log, timeit
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df, config)

@timeit
def transform_datetime(df, config):
    # 将所有时间标签都去掉
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    # 这里hash相当粗糙
    # 将categorical数据映射到一个实数
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    # 这里对于multi_categorical变量的hash更是粗糙
    # 取multi_categorical中第一个值，然后将其映射到一个实数
    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))

@timeit
def data_reduction_train(df):
    matrix = df.as_matrix()
    min_max_scaler = MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)
    pca = PCA()
    pca.fit(matrix)
    sum_ratio, flag_idx = 0, None
    # 确定降维维数
    for i in range(pca.explained_variance_ratio_.size):
        sum_ratio += pca.explained_variance_ratio_[i]
        if sum_ratio >= 0.85:
            flag_idx = i
            break
    if flag_idx:
        pca = PCA(n_components=flag_idx)
        matrix_trans = pca.fit_transform(matrix)
        # reconstruct dataframe
        d = {}
        for i in range(matrix_trans.shape[1]):
            d[f"f_{i}"] = matrix_trans[:,i]
        ret_df = pd.DataFrame(d)
        return ret_df, min_max_scaler, pca

@timeit
def data_reduction_test(df, scaler, pca):
    matrix = df.as_matrix()
    matrix = scaler.transform(matrix)
    matrix_trans = pca.transform(matrix)
    # reconstruct dataframe
    d = {}
    for i in range(matrix_trans.shape[1]):
        d[f"f_{i}"] = matrix_trans[:, i]
    ret_df = pd.DataFrame(d)
    return ret_df