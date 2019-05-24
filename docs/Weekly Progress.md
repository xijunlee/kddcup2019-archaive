# Weekly Progress

## Time Window: 20190415~20190421

- [x] 搭建docker评测环境
- [x] 熟悉baseline模型代码：数据预处理，特征工程，模型参数调优

## Time Window: 20190422~20190428

#### 模型

- [x] 进一步摸清baseline模型
- [x] 学习GBDT

#### 特征工程

- [x] 文献阅读
1. FeatureTools: https://www.kaggle.com/willkoehrsen/automated-feature-engineering-tutorial#Feature-Tools
  
2. Paper: Deep Feature Synthesis: Towards Automating Data Science Endeavors [http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf)
  3. 可能存在的问题：Feature engineering allows us to combine information across many tables into a single dataframe that we can then use for machine learning model training. Finally, the next step after creating all of these features is figuring out which ones are important. 特征太多，不知道哪一个更好。（PCA？非线性降维？）

4. ExploreKit: Automatic Feature Generation and Selection
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7837936&tag=1

#### 超参数调优

- [x] 文献阅读
  
  1. Algorithms for Hyper-Parameter Optimization: <http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>
  
     Sequential model-based global optimization (SMBO):
  
     - Gaussian process (GP): $O(N^{3})$ ，慢

     - Tree-structured Paren estimator (TPE): baseline算法，$O(N)$
     
  2. Multiobjective Neural Network Ensembles Based on Regularized Negative Correlation Learning: <https://ieeexplore.ieee.org/abstract/document/5416712/>
  
     多目标优化方法构建ensamble
  
- [x]  方案探索

  1. 利用超参搜索过程中不同参数训练的模型构建ensamble，无需多余计算量，下期进行实验
  
  

## Time Window: 20190429~20190512

#### 模型

- [x] 学习xgboost，lightGBM

#### 特征工程

- [x] 文献阅读
- [x] 方案探索
  - [x] featuretools：思路类似baseline方法中特征生成方法
  
  - [x] baseline方法特征工程超参改了些超参，直接提交，rank：42
  
  - [x] factorization machine:  y(x)=w_0 + \sum_{i=1}^n w_ix_i + \sum_{i=1}^n\sum_{j=i+1}^n <v_i,v_j> x_i x_j
  
    即抓住任意两个特征之间的interaction产生新的特征，但是这个技术是2010年的了，比较老了。
  
    https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

#### 超参数调优

- [x] 基于高斯过程修改baseline超参数调优模块，然后提交一版

- [x] 文献阅读

- [x] 方案探索
  利用超参搜索过程中不同参数训练的模型构建ensamble，无需多余计算量，进行实验:
  1. 根据AUC，选取最好的5组参数
  
  2. 根据NCL，选取最好的5组参数 (NSGA-II) 
  
     | Algorithms                  | Score        |
     | --------------------------- | ------------ |
     | baseline                    | 0.7121998069 |
     | ensemble_AUC (top5)         | 0.7175696518 |
     | ensemble_NCL (NSGA-II top5) | 0.7185983299 |
  
     
  

## Time Window: 20190513~20190520

特征工程修改思路：
   1. There must exist a relationship between HASH_MAX and WINDOW_SIZE:
      The larger the HASH_MAX, the less information from other records with identical hash value can be used.
      The larger the WINDOW_SIZE, the more temporal information can be used.
      因此会增加四个超参数。
      在merge.py中代码修改如下，已实现，并于20190516晚上提交一版, rank:
      
      ```python
      def temporal_join(u, v, v_name, key, time_col):
          timer = Timer()
      
          window_size = CONSTANT.WINDOW_SIZE if len(u) * 0.001 < CONSTANT.WINDOW_SIZE else int(len(u) * 0.001)
          hash_max = CONSTANT.HASH_MAX if len(u) / CONSTANT.HASH_MAX > 100.0 else int(len(u) / 100.0)
      
          if isinstance(key, list):
              assert len(key) == 1
              key = key[0]
      
          tmp_u = u[[time_col, key]]
          timer.check("select")
      
          tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
          timer.check("concat")
      
          rehash_key = f'rehash_{key}'
          tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % hash_max)
          timer.check("rehash_key")
      
          tmp_u.sort_values(time_col, inplace=True)
          timer.check("sort")
      
          agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                       and not col.startswith(CONSTANT.TIME_PREFIX)
                       and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
      
          tmp_u = tmp_u.groupby(rehash_key).rolling(window_size).agg(agg_funcs)
          timer.check("group & rolling & agg")
      
          tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
          timer.check("reset_index")
      
          tmp_u.columns = tmp_u.columns.map(lambda a:
              f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")
      
          if tmp_u.empty:
              log("empty tmp_u, return u")
              return u
      
          ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
          timer.check("final concat")
      
          del tmp_u
      
          return ret
      ```
   2. 增加PCA降维：将经过各种join后的主表的所有特征输入PCA算法，输出信息占比85%以上的降维特征
   3. 混合PCA降维加上原始特征
   4. 对于原始单列特征增加更多aggregation操作

- [x] 超参数调优修改思路：
   1. 测试class imbalance相关参数
   	a. 固定参数：is_unbalance = True
   	->性能降低
   	b. 自动调参：scale_pos_weight = hp.loguniform('scale_pos_weight', np.log(np.sum(y == 0)/np.sum(y == 1)), 0) 
   	if np.sum(y == 0)/(np.sum(y == 1) + 0.0001) > 1 
   	else hp.loguniform('scale_pos_weight', 0, np.log(np.sum(y == 0)/np.sum(y == 1))),weight
   	->性能降低，但优于固定参数
   2. 文献阅读 
   Efficient and Robust Automated Machine Learning
   http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
   a. 自动超参调优算法：random-forest based SMAC （Baysian optimization），下周进行测试对比TPE
   b. 集成学习选择方法：ensemble selection，下周进行测试对比NCL
   3. 搜集往届获奖方案
   a. https://github.com/flytxtds/AutoGBT (NIPS2018, 1st)
   b. https://github.com/MetaLearners/NIPS-2018-AutoML-Challenge (NIPS2018, 2nd)
   c. https://github.com/jungtaekkim/automl-challenge-2018 (PAKDD2018, 2nd)

# Time Window: 20190521~20190527

特征工程修改思路：
1. random sampling, or data subsampling: 不一定要用到给定数据集的所有数据，resample一些出来学习，提高效率；
    - data ubsampling **(已实现)**
    - data downsampling **(已实现，已提交)**
2. Categorical Feature: an integer describing which category the instance belongs to.
    - preprocessing methods: Hash coding and frequency coding **（frequency coding已实现，效果一般, rank: 37）**
    - advice: avoid OneHot for high cardinality columns and decision tree-based algorithms.
3. 关掉PCA，设计特殊的Feature selection: 需要设计特别的feature selection方法，感觉PCA既耗时，只是选出信息量大的特征，但是没有选出真正有用的特征。
    - select feature according to feature importance generated by lightGBM **(已实现，这个非常有用，rank:29)**
    - feature selection with null importance
        - [https://www.kaggle.com/ogrellier/feature-selection-with-null-importances](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)
        - 这个方法统计特性貌似很好，但是很费时，可以试一下
    - bagging methods: 5-kfolds feature importance average / bagging with different lightGBM model
4. Numerical Feature: a real value.
    - preprocessing methods: standardization
    - For a random variable X, standardization means converting X to its standardized random variable
5. Multi-value Categorical Feature: a set of integers, split by the comma.
    - preprocessing methods: Hash coding and frequency coding
6. Time Feature: an integer describing time information.
    - preprocessing methods: Using second-order features. What is second-order feature ???
7. First-order feature engineer: frequency encoding of categorical features
8. High-order feature engineer:
    - predefine a set of binary transformations based on prior knowledge
    - apply each type of transformation on the original feature sets to generate new features in an expansion-reduction fashion
    - such as:
        - numerical-numerical: +,-,*,/
        - categorical-numerical: num_mean_groupby_cat
        - categorical-categorical: cat_cat_combine, cat_nunique_groupby_cat
        - categorical-temporal: time_difference_groupby_cat
    - key steps:
        1. pre-selection: select features used for feature generation based on prior knowledge
        2. feature generation: generate new feature with all feasible pairs of the pre-selected features
        3. post-selection: select generated features based on the performance and feature importance of a coarsely trained GBDT model
9. Check the imbalance of class: mitigate the imbalance of class
    - Up-sample Minority Class: Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.
    - Down-sample Majority Class: Down-sampling involves randomly removing observations from the majority class to prevent its signal from dominating the learning algorithm.
    - Change Your Performance Metric: For a general-purpose metric for classification, we recommend Area Under ROC Curve (AUROC).
    - Penalize Algorithms (Cost-Sensitive Training): to use penalized learning algorithms that increase the cost of classification mistakes on the minority class. A popular algorithm for this technique is Penalized-SVM
    - Use Tree-Based Algorithms: using tree-based algorithms. Decision trees often perform well on imbalanced datasets because their hierarchical structure allows them to learn signals from both classes.
10. Feature embedding: might utilize DNN to embed the selected feature???