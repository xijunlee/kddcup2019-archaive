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
