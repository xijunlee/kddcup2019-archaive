
NUMERICAL_TYPE = "num"
NUMERICAL_PREFIX = "n_"

CATEGORY_TYPE = "cat"
CATEGORY_PREFIX = "c_"

TIME_TYPE = "time"
TIME_PREFIX = "t_"

MULTI_CAT_TYPE = "multi-cat"
MULTI_CAT_PREFIX = "m_"
MULTI_CAT_DELIMITER = ","


MAIN_TABLE_NAME = "main"
MAIN_TABLE_TEST_NAME = "main_test"
TABLE_PREFIX = "table_"

LABEL = "label"

# There must exist a relationship between HASH_MAX and WINDOW_SIZE:
# 1. The larger the HASH_MAX, the less information from other records with identical hash value can be used.
# 2. The larger the WINDOW_SIZE, the more temporal information can be used.
HASH_MAX = 200
WINDOW_SIZE = 10


# the VARIANCE RAITO is used in PCA
VARIANCE_RATIO = 0.95

REDUCTION_SWITCH = False

FEATURE_SELECTION_SWITCH = True

pre_lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'num_leaves': 127,
        'max_depth': 8,
        'bagging_freq': 1,
        'n_jobs': 4
    }

