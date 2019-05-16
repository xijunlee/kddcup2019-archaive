
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

HASH_MAX = 200

WINDOW_SIZE = 5

# There must exist a relationship between HASH_MAX and WINDOW_SIZE:
# 1. The larger the HASH_MAX, the less information from other records with identical hash value can be used.
# 2. The larger the WINDOW_SIZE, the more temporal information can be used.


