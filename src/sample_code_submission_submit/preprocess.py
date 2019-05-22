import datetime
import CONSTANT
from util import log, timeit
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        # df[c] = df[c].apply(lambda x: int(x))
        df[c], _ = pd.factorize(df[c])
        # Set feature type as categorical
        df[c] = df[c].astype('category')
        # TODO: categorical feature -> encode as frequency

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
        # TODO: multi value categorical feature -> ?
@timeit
def data_reduction_train(df):
    matrix = df.as_matrix()
    min_max_scaler = MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)
    pca = PCA()
    pca.fit(matrix)
    sum_ratio, flag_idx = 0, None
    # determine the reduced dimension
    for i in range(pca.explained_variance_ratio_.size):
        sum_ratio += pca.explained_variance_ratio_[i]
        if sum_ratio >= CONSTANT.VARIANCE_RATIO:
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

@timeit
def feature_selection(X, y, config, seed=None):
    categorical_feats = [
        col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)
    ]
    train_features = X.columns
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(X, y, free_raw_data=False, silent=True)
    lgb_params = CONSTANT.pre_lgb_params
    lgb_params["seed"] = seed
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(X))

    imp_df.sort_values(by=["importance_gain", "importance_split"], ascending=False, inplace=True)

    selected_features = []
    for _, row in imp_df.iterrows():
        if row["importance_gain"] > 0:
            selected_features.append(row["feature"])
        else:
            break
    return X[selected_features], selected_features

def get_feature_importance(df):
    pass