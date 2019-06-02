import datetime
import CONSTANT
from util import log, timeit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE, RFECV
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import category_encoders as ce
import random
import featuretools as ft
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

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
    # for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
    #     df[c].fillna(-1, inplace=True)

    numerical_list = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    df[numerical_list] = SimpleImputer(strategy="median").fit_transform(df[numerical_list])

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)
    # categorical_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    # df[categorical_list] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_list])

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def feature_engineer_ft(df, config):

    es = ft.EntitySet(id='main')
    es = es.entity_from_dataframe(entity_id='main',
                                  dataframe=df,
                                  index="index")
    agg_primitives = CONSTANT.agg_primitives
    trans_primitives = CONSTANT.trans_primitives

    features, feature_names = ft.dfs(
        entityset=es,
        target_entity='main',
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=1,
        n_jobs=1,
        verbose=True)

    features = transform_categorical_hash(features)

    return features

@timeit
def feature_engineer_base(df, config):
    df = transform_categorical_hash(df)
    transform_datetime(df, config)
    return df

@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)
    return df

@timeit
def transform_categorical_hash(df):

    cat_param = CONSTANT.cat_hash_params["cat"]
    multi_cat_param = CONSTANT.cat_hash_params["multi_cat"]

    if cat_param["method"] == "fact":
        # categorical encoding mechanism 1:
        for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
            # df[c] = df[c].apply(lambda x: int(x))
            df[c], _ = pd.factorize(df[c])
            # Set feature type as categorical
            df[c] = df[c].astype('category')
    elif cat_param["method"] == "bd":
        # categorical encoding mechanism 2:
        categorical_feats = [
            col for col in df.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)
        ]

        # Specify the columns to encode then fit and transform
        encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=categorical_feats)
        encoder.fit(df, verbose=1)
        df = encoder.transform(df)
    elif cat_param["method"] == "freq":
        # categorical encoding mechanism 3:
        for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
            # calculate the frequency of item
            val_freq = df[c].value_counts(normalize=True).to_dict()
            df[c] = df[c].map(val_freq)
            df[c] = df[c].astype('float')
    elif cat_param["method"] == "ohe":
        # one hot encoding
        pass

    if multi_cat_param["method"] == 'base':
        for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
    elif multi_cat_param["method"] == 'count':
        for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            df[c] = df[c].apply(lambda x: len(x.split(',')))
    elif multi_cat_param["method"] == 'freq':
        for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
            multi_cat_expand = df[c].astype(str).str.split(',', expand=True).stack() \
                .reset_index(level=0).set_index('level_0').rename(columns={0: c})
            val_freq = multi_cat_expand[c].value_counts(normalize=True).to_dict()
            multi_cat_expand[c] = multi_cat_expand[c].map(val_freq)
            multi_cat_expand[c] = multi_cat_expand[c].astype('float')
            multi_cat_expand = multi_cat_expand.groupby('level_0').agg([sum, np.mean, np.std]).fillna(0)
            multi_cat_expand.columns = multi_cat_expand.columns.map('|'.join).str.strip('|')
            df.drop(columns=c, inplace=True)
            df = pd.concat([df, multi_cat_expand], axis=1)

    # TODO: multi value categorical feature -> ?
    # x = df[c].str.split(r',', expand=True)\
    #         .stack()\
    #         .reset_index(level=1, drop=True)\
    #         .to_frame(c)
    # cleaned = df[c].str.split(r',', expand=True).stack()
    # cleaned = pd.get_dummies(cleaned, prefix='c', columns=c).groupby(level=0).sum()
    # df_r.drop(columns=c, inplace=True)
    # df_r = pd.concat([df_r, cleaned], axis=1)

    return df


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
def data_downsampling(X, y, config, seed=CONSTANT.DOWNSAMPLING_SEED):
    origin_size = len(X)
    X["class"] = y
    df_sampled = resample(X, replace=False, n_samples=int(origin_size*CONSTANT.DOWNSAMPLING_RATIO))
    return df_sampled.drop(columns=["class"]), df_sampled["class"]

@timeit
def data_balance(X, y, config, seed=CONSTANT.DATA_BALANCE_SEED):
    # balance the raw dataset if there exist imbalance class in it.

    origin_size = len(X)
    X["class"] = y
    df_class_0 = X[X["class"]==0]#.drop(columns=["class"])
    df_class_1 = X[X["class"]==1]#.drop(columns=["class"])

    if len(df_class_0) < len(df_class_1):
        df_minority = df_class_0
        df_majority = df_class_1
    else:
        df_minority = df_class_1
        df_majority = df_class_0

    if CONSTANT.SAMPLE_UP_OR_DOWN == "up":
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority))  # to match majority class
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        df_sampled = resample(df_upsampled,
                              replace=False,
                              n_samples=int(origin_size * 0.5),
                              random_state=seed)
    else:
        # Downsample majority class
        n_sample = int(0.16*len(df_majority)) if len(df_majority) > 6*len(df_minority) \
            else int(0.3*len(df_majority))
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample without replacement
                                           n_samples=n_sample,
                                           random_state=seed)  # to match minority class

        # Combine minority class with downsampled majority class
        df_sampled = pd.concat([df_majority_downsampled, df_minority])


    # Display new class counts
    print(df_sampled["class"].value_counts())

    return df_sampled.drop(columns=["class"]), df_sampled["class"]


@timeit
def feature_generation(X, random_features=None, seed=None):
    # Unary operation
    for c in [c for c in X if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        # discretization
        c_min = int(X[c].min())
        segment = (X[c].max() - c_min) / 10
        X = X.assign(disc=lambda x: (x[c] - c_min) // segment * segment + c_min)
        X = X.rename(columns={'disc': c+'_disc'})

    # Binary operation
    if random_features is None:
        random_feature_1 = random.sample([c for c in X if c.startswith(CONSTANT.NUMERICAL_PREFIX)], 20)
        random_feature_2 = random.sample([c for c in X if c.startswith(CONSTANT.NUMERICAL_PREFIX)], 20)
    else:
        random_feature_1, random_feature_2 = random_features

    for c_1 in random_feature_1:
        for c_2 in random_feature_2:
            X[c_1 + '_plus_' + c_2] = X[c_1] + X[c_2]
            X[c_1 + '_minus_' + c_2] = X[c_1] - X[c_2]
            X[c_1 + '_multiple_' + c_2] = X[c_1] * X[c_2]
            X[c_1 + '_divide_' + c_2] = X[c_1] / (X[c_2] + 1e-8)

    if random_features is None:
        return X, [random_feature_1, random_feature_2]
    else:
        return X

@timeit
def feature_selection(X_raw, y_raw, config, seed=CONSTANT.FEATURE_SELECTION_SEED):

    method = CONSTANT.feature_selection_param["method"]
    X, y = data_balance(X_raw, y_raw, config)
    X, y = data_downsampling(X, y, config)
    selected_features = []
    if method == "imp":
        selected_features = _imp_feature_selection(X, y, config, seed)
    elif method == "nh":
        selected_features = _nh_feature_selection(X, y, config, seed)
    elif method == "rfe":
        selected_features = _rfe_feature_selection(X, y, config, seed)
    elif method == "sfm":
        selected_features =_sfm_feature_selection(X, y, config, seed)
    elif method =="cor":
        selected_features = _cor_feature_selection(X, y, config, seed)
    return X_raw[selected_features], selected_features

@timeit
def _cor_feature_selection(X_raw, y_raw, config, seed=None):
    cor_list = []
    # calculate the correlation with y for each feature
    feature_name = X_raw.columns.tolist()
    for i in feature_name:
        cor = np.corrcoef(X_raw[i], y_raw)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    n_selected_ratio = 0.2
    n_selection_feature = int(0.2*len(feature_name))
    cor_feature = X_raw.iloc[:, np.argsort(np.abs(cor_list))[-1*n_selection_feature:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_feature


@timeit
def _sfm_feature_selection(X_raw, y_raw, config, seed=None):
    # X, y = data_downsampling(X_raw, y_raw, config)
    X, y = X_raw, y_raw

    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
    embeded_lgb_selector.fit(X, y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:, embeded_lgb_support].columns.tolist()
    # print(str(len(embeded_lgb_feature)), 'selected features')
    return embeded_lgb_feature

@timeit
def _rfe_feature_selection(X_raw, y_raw, config, seed=None):
    '''
    select feature using recursive feature elimination method
    :param X_raw:
    :param y_raw:
    :param config:
    :param seed:
    :return:
    '''
    X, y = data_downsampling(X_raw, y_raw, config)

    if CONSTANT.cat_hash_params["cat"]["method"] == "fact":
        categorical_feats = [
            col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)
        ]
    else:
        categorical_feats = []

    train_features = X.columns
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(X, y, free_raw_data=False, silent=True)
    lgb_params = CONSTANT.pre_lgb_params
    lgb_params["seed"] = seed
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, categorical_feature=categorical_feats)

    # create the RFE model and select feature
    rfe = RFE(clf)
    rfe = rfe.fit(X, y)
    print(rfe.support_)

    # summarize the ranking of the features
    feature_rank = pd.DataFrame({"feature": X.columns, "rank": rfe.ranking_})
    feature_rank = feature_rank.sort_values(by=["rank"], ascending=True)

    print(feature_rank)

@timeit
def _imp_feature_selection(X_raw, y_raw, config, seed=None):
    '''
    select feature based on feature importance
    :param X_raw:
    :param y_raw:
    :param config:
    :param seed:
    :return:
    '''

    # X, y = data_downsampling(X_raw, y_raw, config)
    X, y = X_raw, y_raw
    # if CONSTANT.cat_hash_params["cat"]["method"] == "fact":
    #     categorical_feats = [
    #         col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)
    #     ]
    # else:
    #     categorical_feats = []

    # categorical_feats = [c for c in X.columns if X[c].dtype == "object"]

    train_features = X.columns
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(X, y, free_raw_data=False, silent=True)
    lgb_params = CONSTANT.pre_lgb_params
    lgb_params["seed"] = seed
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)
    # if there still exist categorical features
    #clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(X))

    # imp_df.sort_values(by=["importance_gain"], ascending=False, inplace=True)

    selected_features = []
    selected_features = imp_df.query("importance_gain > 0")["feature"]

    return selected_features

@timeit
def _nh_feature_selection(X_raw, y_raw, config, n_fold=10, seed=None):

    '''
    select feature based on null hypothesis
    :param X_raw:
    :param y_raw:
    :param config:
    :param seed:
    :return:
    '''

    # X, y = data_balance(X_raw, y_raw, config)
    X, y = X_raw, y_raw
    def get_feature_importances(X, y, shuffle, seed=None):
        # Gather real features
        train_features = X.columns
        # Go over fold and keep track of CV score (train and valid) and feature importances
        # Shuffle target if required
        yy = y.copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            yy = y.copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(X, yy, free_raw_data=False, silent=True)
        lgb_params = CONSTANT.pre_lgb_params

        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['trn_score'] = roc_auc_score(yy, clf.predict(X))

        return imp_df

    actual_imp_df = get_feature_importances(X, y, shuffle=False)

    null_imp_df = pd.DataFrame()
    nb_runs = n_fold
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(X, y, shuffle=True, seed=seed)
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(
            1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(
            1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score']) \
        .sort_values(by=['gain_score', 'split_score'], ascending=False)

    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score']) \
        .sort_values(by=['gain_score', 'split_score'], ascending=False)

    selected_features = []
    selected_features = corr_scores_df.query("split_score > 0")["feature"]

    return selected_features

