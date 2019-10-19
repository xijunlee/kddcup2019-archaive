# KDDCup2019

This is an example KDDCup2019 starting kit
==========================================

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
4PARADIGM, CHALEARN, AND/OR OTHER ORGANIZERS
OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

Contents:
---------

ingestion_program/: The code and libraries used on Codalab to run your submmission.
scoring_program/: The code and libraries used on Codalab to score your submmission.
sample_code_submission/: An example of code submission you can use as template.
sample_data/: Some sample data to test your code before you submit it.
sample_ref/: Reference data required to evaluate your submission.

Make the submmission:
---------------------

To make your own submission, follow the instructions:
1. modify sample_code_submission/ to your code (**you should keep metadata file unchanged in sample_code_submission/**)
2. Zip the contents of sample_code_submission (without the directory structure)
```
zip mysubmission.zip sample_code_submission/*
```
3. submit to Codalab competition "Participate>Submit/View results".


Local development and testing:
------------------------------

You can test your code in the exact same environment as the Codalab environment using docker.
You are able to run the ingestion program (to produce predictions) and the scoring program
(to evaluate your predictions) on toy sample data.

Quick Start
-----------

1. If you are new to docker, install docker from https://docs.docker.com/get-started/.
2. At the shell, change to the startingkit directory, run

```
docker run -it --rm -u root -v $(pwd):/app/kddcup codalab/codalab-legacy:py3 bash
```

3. Now your are in the bash of the docker container, run ingestion program

```
cd /app/kddcup
python3 ingestion_program/ingestion.py
```
It runs sample_code_submission and the predictions will be in sample_predictions directory

4. Now run scoring program:

```
python3 scoring_program/score.py
```

It will score the predictions and the results will be in sample_scoring_output directory

### Remark

- you can put public data in sample_data and test it
- The full call of the ingestion program is:

```
python3 ingestion_program/ingestion.py local sample_data sample_predictions ingestion_program sample_code_submission
```

- The full call of the scoring program is:

```
python3 scoring_program/score.py local sample_predictions sample_ref sample_scoring_output
```

题目特点
在这次比赛中，主要有以下难点：

1.挖掘有效的特征

与传统数据挖掘竞赛不同的是，AutoML竞赛中，参赛选手只知道数据的类型（数值变量、分类变量、时间变量、多值分类变量等），而不知道数据的含义，这毫无疑问会增加特征工程的难度，如何挖掘到有效的通用特征成为一个难点。

2.赛题数据和时序相关

时序相关数据的数据挖掘难度较大，在传统的机器学习应用中，需要经验丰富的专家才能从时序关系型数据中挖掘出有效的时序信息，并加以利用提升机器学习模型的效果。即使具备较深的知识储备，专家也需要通过不断的尝试和试错，才能构建出有价值的时序特征，并且利用好多个相关联表来提升机器学习模型的性能。

3.赛题数据按照多表给出

赛题的数据是按照多表给出的，这就要求参赛选手能够构建一个处理多表之间多样的连接关系的自动化机器学习系统。多表数据无疑提升了对系统的稳定性的要求，稍有不慎，有可能合并出来的数据过于庞大就直接超时或者超内存而导致没有最终成绩。

4.时间内存限制严格

比赛代码运行环境是一个4核CPU，16G内存的docker环境，对于未知大小的数据集，在代码执行过程中的某些操作很容易使得内存峰值超过16G，导致程序崩溃。因此选手要严格优化某些操作，或使用采样等方式完成任务。此外，比赛方对于每个数据集严格限制了代码的执行时间，稍有不慎就会使得运行时间超时而崩溃。

解决方案
我们团队基于所给数据实现了一套支持多表的AutoML框架，包括自动多表合并、自动特征工程、自动特征选择、自动模型调参、自动模型融合等步骤，在时间和内存的控制上我们也做了很多优化工作。

{%img http://xijun-album.oss-cn-hangzhou.aliyuncs.com/alphagozero/WechatIMG118.png %}

数据预处理