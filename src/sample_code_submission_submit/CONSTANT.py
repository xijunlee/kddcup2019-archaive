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
VARIANCE_RATIO = 0.95  # the VARIANCE RAITO is used in PCA

num_generate_order = 2
num_primitives = [
    "cum_mean",
    "cum_sum",
    "cum_max",
    "cum_min",
    # "cum_prod",
]

time_primitives = [
    # "year",
    # "month",
    "day",
    "hour",
    "minute",
    "second"
]

# Switch and parameter of feature selection
FEATURE_SELECTION_SWITCH = True
FEATURE_RATIO_1 = 0.4
FEATURE_RATIO_2 = 0.7
pre_lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 50,
        'max_depth': 8,
        'bagging_freq': 1,
        'n_jobs': 4,
        'verbose':-1
}
feature_selection_param = {
    "method": "imp" # 4 options: "imp", "nhp", "shap", "sfm", "cor", "chi"
                    # "imp" for feature importance,
                    # "nh" for null hypothesis,
                    # "shap" for shapely value,
                    # "sfm" for SelectFromModel
                    # "cor" for correlation
                    # "chi" for Chi-2
}

# Switch and parameter of data balance
DATA_BALANCE_SWITCH = False
SAMPLE_UP_OR_DOWN = "down"

# Use Bayesian optimization
BAYESIAN_OPT = False

# Drop outliers from the training set
DROP_OUTLIER = False

# Use double validation sets
DOUBLE_VAL = False

# Switch and parameter of data downsampling
DATA_DOWNSAMPLING_SWITCH = False
DOWNSAMPLING_RATIO = 0.5

# Parameter of model ensemble
ENSEMBLE = True
ENSEMBLE_OBJ = 2  # currently 2 is better than 3
ENSEMBLE_SIZE = 5
HPO_EVALS = 30
AUTO = True
STACKING = False
STACKING_METHOD = 1  # 0: lr; 1: lgb
STOCHASTIC_CV = False
TRAIN_DATA_SIZE = 30000


# Parameter of automl
train_lgb_params = {
        "objective": "binary",
        "boosting_type": "rf",
        "metric": "auc",
        "verbosity": -1,
        "seed": None,
        "num_threads": 4,
        'n_jobs': 4,
        # "is_unbalance": True
}

'''
All kinds of seed
'''
SEED = 1
HYPEROPT_SEED = SEED
DOWNSAMPLING_SEED = SEED
DATA_BALANCE_SEED = SEED
FEATURE_SELECTION_SEED = SEED
DATA_SPLIT_SEED = SEED

N_THREAD = 4