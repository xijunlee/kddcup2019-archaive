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
  - [ ] 特征工程改用featuretools，然后提交一版

#### 超参数调优

- [ ] 基于高斯过程修改baseline超参数调优模块，然后提交一版

- [ ] 文献阅读
- [ ] 方案探索
  1. 利用超参搜索过程中不同参数训练的模型构建ensamble，无需多余计算量，进行实验
