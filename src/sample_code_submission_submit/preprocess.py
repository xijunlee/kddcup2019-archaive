import datetime
import CONSTANT
from util import log, timeit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import random
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# import shap
from multiprocessing import Pool
import time
import os
import re

def test_data_feature_selection(X_test, train_selected_feature):

    set_test_feature = set(X_test.columns)
    set_train_feature = set(train_selected_feature)
    set_diff_feature = set_train_feature - set_test_feature
    for col in set_diff_feature:
        pattern = re.compile(r"c_(.*)_DUMMY\((.*)\)")
        m = pattern.match(col)
        feat_name = m.group(1)
        cat_name = m.group(2)
        f = lambda x: 1 if x == cat_name else 0
        X_test[col] = X_test[feat_name].map(f).rename(col)

class MissingValueProcessor:
    def __init__(self):
        self.missing_ratio_threshold = 0.7
        self.mv_dict = {}

    def time_missing_filter(self, df, time_feature_list):
        mask = df[time_feature_list].isnull().any(axis=1) == False
        return df[mask]

    def feature_filter(self, df):
        feature_name = df.columns
        missing_ratio = df.isnull().sum(axis=0).values.astype('float') / df.shape[0]
        df_feature_missing = pd.DataFrame({"feature_name": feature_name, "missing_ratio": missing_ratio})
        drop_features = df_feature_missing.query(f"missing_ratio > {self.missing_ratio_threshold}")["feature_name"].values
        return drop_features

    def num_fit_transform(self, col):
        col.fillna(col.mean())

    def cat_fit_transform(self, col):
        col.fillna(col.mode()[0])

    def mv_fit_transform(self, col):
        col.fillna("0")

    def time_fit_transform(self, col):
        col.fillna(datetime.datetime(1970,1,1))

    def mv_count(self, col):
        pass

class CATEncoder:
    def __init__(self, max_cat_num=10, cum_ratio_thr=0.5):
        self.max_cat_num = max_cat_num
        self.cum_ratio_thr = cum_ratio_thr

    def fit_transform(self, col):

        ret, _ = pd.factorize(col)
        ret_fact = pd.Series(ret, dtype="category", name=col.name)
        # ret_fact = col.astype("category")
        cat_count = ret_fact.value_counts().to_dict()
        val_freq = ret_fact.value_counts(normalize=True).to_dict()
        ret_freq = ret_fact.map(val_freq).rename(f"n_FREQ({col.name})").astype('float')

        n = min(self.max_cat_num, len(cat_count)-1)

        cat_list = list(cat_count.keys())[:n+1]
        f = lambda x: x if x in cat_list[:n] else cat_list[n]
        dummy_col = ret_fact.map(f)

        name_map = lambda x: f"c_{col.name}_DUMMY({x})"
        dummy_cols = pd.get_dummies(dummy_col).rename(name_map, axis=1)
        # ret_cat = pd.concat([dummy_cols, ret_fact, ret_freq], axis=1)
        ret_cat = pd.concat([dummy_cols, ret_fact, ret_freq], axis=1)

        return ret_cat

def seperate(x):
    try:
        x = tuple(x.split(','))
    except AttributeError:
        x = ('-1', )
    return x

class MVEncoder:

    def __init__(self, max_cat_num=600):
        self.max_cat_num = max_cat_num

    def encode(self, cats):
        return min((self.mapping[c] for c in cats))

    def fit_transform(self, col):

        col = col.map(seperate)

        cat_count = {}
        for cats in col:
            for c in cats:
                try:
                    cat_count[c] += 1
                except KeyError:
                    cat_count[c] = 1
        cat_list = np.array(list(cat_count.keys()))
        cat_num = np.array(list(cat_count.values()))
        idx = np.argsort(-cat_num)
        cat_list = cat_list[idx]

        self.mapping = {}
        for i, cat in enumerate(cat_list):
            self.mapping[cat] = min(i, self.max_cat_num)
        del cat_count, cat_list, cat_num

        col_encode = col.map(self.encode)
        col_encode.name = f"c_MVCODE({col.name})"
        col_encode = col_encode.astype("category")
        del col

        return col_encode

class NUMGenerator:
    def __init__(self):
        pass

    def cum_sum(self, col):
        ret = col.cumsum()
        ret.name = f"n_CUMSUM({col.name})"
        del col
        return ret

    def cum_mean(self, col):
        num = np.array([i+1 for i in range(len(col))])
        ret = col.cumsum() / num
        ret.name = f"n_CUMMEAN({col.name})"
        del col
        return ret

    def cum_max(self, col):
        ret = col.cummax()
        ret.name = f"n_CUMMAX({col.name})"
        del col
        return ret

    def cum_min(self, col):
        ret = col.cummin()
        ret.name = f"n_CUMMIN({col.name})"
        del col
        return ret

    def cum_prod(self, col):
        ret = col.cumprod()
        ret.name = f"n_CUMPROD({col.name})"
        del col
        return ret

class TIMEGenrator:
    def __init__(self):
        pass

    def year(self, col):
        ret = col.dt.year
        ret.name = f"n_YEAR({col.name})"
        del col
        return ret

    def month(self, col):
        ret = col.dt.month
        ret.name = f"n_MONTH({col.name})"
        del col
        return ret

    def day(self, col):
        ret = col.dt.day
        ret.name = f"n_DAY({col.name})"
        del col
        return ret

    def hour(self, col):
        ret = col.dt.hour
        ret.name = f"n_HOUR({col.name})"
        del col
        return ret

    def minute(self, col):
        ret = col.dt.minute
        ret.name = f"n_MINUTE({col.name})"
        del col
        return ret

    def second(self, col):
        ret = col.dt.second
        ret.name = f"n_SECOND({col.name})"
        del col
        return ret


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])

@timeit
def clean_df(df):
    # return fillna_rewrite(df)
    fillna_rewrite_seq(df)

@timeit
def fillna_rewrite_seq(df):

    # delete row whose time is nan
    time_feature_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    # if len(time_feature_list):
    #     mask = df[time_feature_list].isnull().any(axis=1)
    #     df.drop(df[mask].index, inplace=True)

    # filter feature whose raito not nan < 0.7
    missing_ratio_threshold = 0.7
    feature_name = df.columns
    missing_ratio = df.isnull().sum(axis=0).values.astype('float') / df.shape[0]
    df_feature_missing = pd.DataFrame({"feature_name": feature_name, "missing_ratio": missing_ratio})
    drop_feature = df_feature_missing.query(f"missing_ratio > {missing_ratio_threshold}")["feature_name"].values
    df.drop(columns=drop_feature, inplace=True)

    num_feature_list = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_feature_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    mul_feature_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    time_feature_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    for col in num_feature_list:
        # mean = df[col].mean()
        df[col].fillna(df[col].mean(), inplace=True)

    for col in cat_feature_list:
        # mode = df[col].mode()[0]
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = df[col].astype("category")


    for col in mul_feature_list:
        df[col].fillna("-1", inplace=True)

    for col in time_feature_list:
        df[col].fillna(datetime.datetime(1970, 1, 1), inplace=True)

@timeit
def fillna_rewrite(df):
    mvp = MissingValueProcessor()
    time_feature_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    # if len(time_feature_list):
    #     df = mvp.time_missing_filter(df, time_feature_list)

    drop_features = mvp.feature_filter(df)
    df.drop(columns=drop_features, inplace=True)

    num_feature_list = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_feature_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    mul_feature_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]

    with Pool(processes=CONSTANT.N_THREAD) as pool:
        pool.map(mvp.num_fit_transform, [df[col] for col in num_feature_list])
        pool.close()
        pool.join()
    with Pool(processes=CONSTANT.N_THREAD) as pool:
        pool.map(mvp.cat_fit_transform, [df[col] for col in cat_feature_list])
        pool.close()
        pool.join()

    for col in cat_feature_list:
        df[col].astype('category')

    with Pool(processes=CONSTANT.N_THREAD) as pool:
        pool.map(mvp.mv_fit_transform, [df[col] for col in mul_feature_list])
        pool.close()
        pool.join()
    with Pool(processes=CONSTANT.N_THREAD) as pool:
        pool.map(mvp.time_fit_transform, [df[col] for col in time_feature_list])
        pool.close()
        pool.join()

    # time_list = [df[col] for col in time_feature_list]

    # ret = pd.concat(num_list + cat_list + mul_list + time_list, axis=1)

    # del df, num_list, cat_list, mul_list, time_list
    # return ret

@timeit
def fillna(df):

    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)


    numerical_list = [MissingValueProcessor for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]

    for c in numerical_list:
        df[c] = mvp.fit_transform(df[c], "num", "mean")

    # all_mv = pd.DataFrame()
    with Pool(processes=2) as pool:
        df_num = pool.map(SimpleImputer(strategy="mean").fit_transform, [df[col] for col in numerical_list])
        all_mv = pd.concat(pd.DataFrame(df_num), axis=1)
        pool.close()
        pool.join()

    # df[numerical_list] = SimpleImputer(strategy="mean").fit_transform(df[numerical_list])

    # for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
    #     df[c].fillna("0", inplace=True)
    categorical_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    df[categorical_list] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_list])

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)
    # time_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]
    # df[time_list] = SimpleImputer(strategy="most_frequent").fit_transform(df[time_list])

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)

@timeit
def feature_engineer_rewrite_seq(df, config):
    df.reset_index(inplace=True, drop=True)
    print(f"length of data: {len(df)}")

    num_feature_list = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_feature_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    mul_feature_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    time_feature_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    catEncoder = CATEncoder()
    for col in cat_feature_list:
        df = df.join(catEncoder.fit_transform(df[col]))
        df = df.join(catEncoder.factorize(df[col]))
    print(df)

@timeit
def feature_engineer_rewrite(df, config):

    df.reset_index(inplace=True, drop=True)
    print(f"length of data: {len(df)}")

    num_feature_list = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_feature_list = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    mul_feature_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    # time_feature_list = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    feat_list = []

    # process category feature
    st_time = time.time()
    catEncoder = CATEncoder()
    with Pool(processes=CONSTANT.N_THREAD) as pool:
        feat_list += pool.map(catEncoder.fit_transform, [df[col] for col in cat_feature_list])
        pool.close()
        pool.join()
    ed_time = time.time()
    print(f"duration of catEncoder.fit_transform: {ed_time-st_time}")

    # process multi value feature
    # st_time = time.time()
    # mveEncoder = MVEncoder()
    # with Pool(processes=CONSTANT.N_THREAD) as pool:
    #     feat_list += pool.map(mveEncoder.fit_transform, [df[col] for col in mul_feature_list])
    #     pool.close()
    #     pool.join()
    # ed_time = time.time()
    # print(f"duration of mveEncoder.fit_transform: {ed_time-st_time}")

    # process number feature
    numGenerator = NUMGenerator()
    funcs, num_generate_order = CONSTANT.num_primitives, CONSTANT.num_generate_order
    order_feature_list = [[] for _ in range(num_generate_order)]
    order_feature_list[0] = [df[col] for col in num_feature_list]
    for i in range(1, num_generate_order):
        for func in funcs:
            st_time = time.time()
            with Pool(processes=CONSTANT.N_THREAD) as pool:
                order_feature_list[i] += pool.map(getattr(numGenerator, func), order_feature_list[i-1])
                pool.close()
                pool.join()
            ed_time = time.time()
            print(f"duration of numGenerator.{func}: {ed_time-st_time}")
    for order_feature in order_feature_list:
        feat_list += order_feature

    timeGenerator = TIMEGenrator()
    funcs, time_list = CONSTANT.time_primitives, []
    for func in funcs:
        st_time = time.time()
        cmd = "timeGenerator." + func + "(df[config[\"time_col\"]])"
        feat_list += [eval(cmd)]
        ed_time = time.time()
        print(f"duration of timeGenerator.{func}: {ed_time-st_time}")

    # ret = pd.concat(cat_list + num_list + time_list, axis=1)

    return pd.concat(feat_list, axis=1)


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

    # process the categorical feature
    features = transform_categorical_hash(features)
    # replace year 1970 with the most frequently emerge year
    for c in [c for c in features if c.startswith("YEAR")]:
        mode = features[c].mode()[0]
        features[c].replace(to_replace=1970, value=mode, inplace=True)


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
            df[c], _ = pd.factorize(df[c])
            # Set feature type as categorical
            df[c] = df[c].astype('category')

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
    # origin_size = len(X)
    X["class"] = y
    len_sample_1 = len(X[X["class"] == 1])
    len_sample_0 = len(X[X["class"] == 0])
    df_sampled_1 = resample(X[X["class"] == 1], replace=False,
                          n_samples=int(len_sample_1 * CONSTANT.DOWNSAMPLING_RATIO),
                          random_state=seed)
    df_sampled_0 = resample(X[X["class"] == 0], replace=False,
                            n_samples=int(len_sample_0 * CONSTANT.DOWNSAMPLING_RATIO),
                            random_state=seed)

    # return df_sampled.drop(columns=["class"]), df_sampled["class"]
    df_sampled = pd.concat([df_sampled_0, df_sampled_1], axis=0)

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
def feature_selection(X_raw, y_raw, config, n_selected_ratio, seed=CONSTANT.FEATURE_SELECTION_SEED):

    print("before feature selection")

    method = CONSTANT.feature_selection_param["method"]
    feature_name = X_raw.columns.tolist()
    len_X_01 = int(len(X_raw) * 0.1)
    len_feature = len(feature_name)
    # n_selected_feature = len_X_01 if len_X_01 >= len_feature else int(n_selected_ratio * len(feature_name))
    n_selected_feature = int(n_selected_ratio * len(feature_name))


    # if CONSTANT.DATA_BALANCE_SWITCH:
    #     X, y = data_balance(X_raw, y_raw, config)
    if CONSTANT.DATA_DOWNSAMPLING_SWITCH:
        X, y = data_downsampling(X_raw, y_raw, config)
    X, y = X_raw, y_raw
    selected_features = []

    if method == "imp":
        selected_features = _imp_feature_selection(X, y, config, n_selected_feature, seed)
    elif method == "nh":
        selected_features = _nh_feature_selection(X, y, config, seed)
    elif method == "shap":
        selected_features = _shap_feature_selection(X, y, config, n_selected_feature, seed)
    elif method == "sfm":
        selected_features =_sfm_feature_selection(X, y, config, seed)
    elif method == "cor":
        selected_features = _cor_feature_selection(X, y, config, n_selected_feature, seed)
    elif method == "chi":
        selected_features = _chi_feature_selection(X, y, config, n_selected_feature, seed)
    print(f"Selected {n_selected_feature} features")
    return X_raw[selected_features], selected_features

@timeit
def _chi_feature_selection(X_raw, y_raw, config, n_selected_feature, seed=None):

    X_norm = MinMaxScaler().fit_transform(X_raw)
    chi_selector = SelectKBest(chi2, k=n_selected_feature)
    chi_selector.fit(X_norm, y_raw)
    chi_support = chi_selector.get_support()
    chi_feature = X_raw.loc[:, chi_support].columns.tolist()
    return chi_feature

@timeit
def _cor_feature_selection(X_raw, y_raw, config, n_selected_feature, seed=None):
    cor_list = []
    # calculate the correlation with y for each feature
    feature_name = X_raw.columns.tolist()
    for i in feature_name:
        cor = np.corrcoef(X_raw[i], y_raw)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X_raw.iloc[:, np.argsort(np.abs(cor_list))[-1*n_selected_feature:]].columns.tolist()
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
def _shap_feature_selection(X_raw, y_raw, config, n_selected_features, seed=None):
    # Create train and validation set
    train_x, valid_x, train_y, valid_y = train_test_split(X_raw, y_raw, test_size=0.2, shuffle=True, stratify=y_raw,
                                                          random_state=seed)
    train_features = X_raw.columns
    lgb_params = CONSTANT.pre_lgb_params
    lgb_params["seed"] = seed
    dtrain = lgb.Dataset(train_x, train_y, free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(valid_x, valid_y, free_raw_data=False, silent=True)
    lgbm = lgb.train(lgb_params,
                     dtrain,
                     2500,
                     valid_sets=dvalid,
                     early_stopping_rounds=30,
                     verbose_eval=10
                     )

    shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([train_features, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    selected_features = importance_df.iloc[:n_selected_features, 0]
    selected_features_gt_zero = importance_df.query("shap_importance > 0")["column_name"]

    if len(selected_features_gt_zero) < n_selected_features:
        return selected_features_gt_zero

    return selected_features

@timeit
def _imp_feature_selection(X_raw, y_raw, config, n_selected_features, seed=None):
    '''
    select feature based on feature importance
    :param X_raw:
    :param y_raw:
    :param config:
    :param seed:
    :return:
    '''

    X, y = X_raw, y_raw

    train_features = X.columns.values
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(X, y, free_raw_data=False, silent=True)
    lgb_params = CONSTANT.pre_lgb_params
    lgb_params["seed"] = seed
    lgb_params["colsample_bytree"] = np.sqrt(len(train_features)) / len(train_features)
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)
    # if there still exist categorical features
    #clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = train_features
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(X))

    imp_df.sort_values(by=["importance_gain"], ascending=False, inplace=True)

    selected_features = []
    selected_features_gt_zero = imp_df.query("importance_gain > 0")["feature"]
    selected_features = imp_df.iloc[:n_selected_features, 0]

    if len(selected_features_gt_zero) < n_selected_features:
        return selected_features_gt_zero

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

