# Weekly Progress

## Time Window: 20190415~20190421

- [x] 搭建docker评测环境
- [x] 熟悉baseline模型代码：数据预处理，特征工程，模型参数调优

## Time Window: 20190422~20190428

#### 模型

- [x] 进一步摸清baseline模型
- [ ] 学习GBDT

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

- [ ] 学习xgboost，lightGBM

#### 特征工程

- [ ] 文献阅读
- [ ] 方案探索
  - [x] featuretools：思路类似baseline方法中特征生成方法
  
  - [x] baseline方法特征工程超参改了些超参，直接提交，rank：42
  
  - [x] factorization machine:  y(x)=w_0 + \sum_{i=1}^n w_ix_i + \sum_{i=1}^n\sum_{j=i+1}^n <v_i,v_j> x_i x_j
  
    即抓住任意两个特征之间的interaction产生新的特征，但是这个技术是2010年的了，比较老了。
  
    https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

#### 超参数调优

- [ ] 基于高斯过程修改baseline超参数调优模块，然后提交一版

- [ ] 文献阅读

- [ ] 方案探索
  利用超参搜索过程中不同参数训练的模型构建ensamble，无需多余计算量，进行实验:
  1. 根据AUC，选取最好的5组参数
  
  2. 根据NCL，选取最好的5组参数 (NSGA-II)
  
     | Algorithms                  | Score        |
     | --------------------------- | ------------ |
     | baseline                    | 0.7121998069 |
     | ensemble_AUC (top5)         | 0.7175696518 |
     | ensemble_NCL (NSGA-II top5) | 0.7185983299 |
  
     
  

## Time Window: 20190513~20190520

- [x] 特征工程修改思路：
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
      


