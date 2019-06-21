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

# parameter of table merge
'''
There must exist a relationship between HASH_MAX and WINDOW_SIZE:
1. The larger the HASH_MAX, the less information from other records with identical hash value can be used.
2. The larger the WINDOW_SIZE, the more temporal information can be used.
'''
HASH_MAX = 200
HASH_BIN = 100
WINDOW_SIZE = 5
WINDOW_RATIO = 0.001

# Switch and parameter of data reduction
REDUCTION_SWITCH = False
VARIANCE_RATIO = 0.95 # the VARIANCE RAITO is used in PCA

# Switch and parameter of feature generation
FEATURE_GENERATION_SWITCH = False

# Swich and parameter of feature generation using featuretools
FEATURE_ENGINEERING_FT_SWITCH = True
# Select the agg and trans primitives you want to look over
agg_primitives=[
        # 'std', 'min', 'max', 'mean',
        # 'percent_true', 'last', 'count',
        # 'trend', 'n_most_common'
]
trans_primitives=[
        'cum_mean', 'cum_min', 'cum_max',
        'hour', 'week', 'month', 'year'
        #'percentile', 'cum_mean', 'cum_min', 'cum_count', 'cum_max'
        # 'subtract_numeric', 'add_numeric', 'diff', 'absolute',
        # 'modulo_numeric', 'hour', 'week', 'month', 'second', 'minute', 'weekday', 'year'
]

# Swich and parameter of feature generation using basing method
FEATURE_ENGINEERING_BASE_SWITCH = False if FEATURE_ENGINEERING_FT_SWITCH else True

# Switch and parameter of feature selection
FEATURE_SELECTION_SWITCH = True
pre_lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 100,
        'max_depth': 8,
        'bagging_freq': 1,
        'n_jobs': 4,
        'verbose':-1
}
feature_selection_param = {
    "method": "chi" # 4 options: "imp", "nhp", "rfe", "sfm", "cor", "chi"
                    # "imp" for feature importance,
                    # "nh" for null hypothesis,
                    # "rfe" for recursive feature elimination,
                    # "sfm" for SelectFromModel
                    # "cor" for correlation
                    # "chi" for Chi-2
}

# Switch and parameter of data balance
DATA_BALANCE_SWITCH = False
SAMPLE_UP_OR_DOWN = "down"


BAYESIAN_OPT = False
# Switch and parameter of data downsampling
DATA_DOWNSAMPLING_SWITCH = False
DOWNSAMPLING_RATIO = 0.5

# Parameter of model ensemble
ENSEMBLE = True
ENSEMBLE_OBJ = 2  # currently 2 is better than 3
ENSEMBLE_SIZE = 5
HPO_EVALS = 10
AUTO = True
STACKING = False
STACKING_METHOD = 1  # 0: lr; 1: lgb
STOCHASTIC_CV = False
TRAIN_DATA_SIZE = 30000

# Parameter of categorical hash
cat_hash_params = {
    "cat": {
        "method": "fact" # 3 options : "bd", "freq", "fact"
    },
    "multi_cat": {
        "method": "count" # 3 options: "freq", "count", "base"
    }
}

# Parameter of automl
train_lgb_params = {
        "objective": "binary",
        # 'boosting_type': 'rf',
        "metric": "auc",
        "verbosity": -1,
        "seed": None,
        "num_threads": 4,
        # "is_unbalance": True
}

'''
All kinds of seed
'''
# HYPEROPT_SEED = None
# DOWNSAMPLING_SEED = None
# DATA_BALANCE_SEED = None
# FEATURE_SELECTION_SEED = None
SEED = 1
HYPEROPT_SEED = SEED
DOWNSAMPLING_SEED = SEED
DATA_BALANCE_SEED = SEED
FEATURE_SELECTION_SEED = SEED
DATA_SPLIT_SEED = SEED