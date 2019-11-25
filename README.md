# MSBD5001 Individual Project

Thanks to professors and TA for arranging this individual project. In this project, we are supposed to predict the gaming hours according to the given data. Shortage of training data is the major problem of this project and the labels are very sparse, most of them are closed to zero.

I pick 2 models to do prediction since there are 2 results can be used in kaggle. The first one is xgboosting, a tree based model, quite popular among Kaggle projects. The second one is adaboosting, a bagging model.

I'm perssimistic about the final result due to the shortage of training data, the models are easily to get overfitting. But I think the final result isn't everthing, the key point of this project should be applying the standard process of modeling and enjoying data analysis.

Based on what I have learnt from MSBD5001, here are major steps for this project:

- Data Cleansing
- Feature Selection
- Training Data Preparation
- Modeling
- Tuning
- Prediction

## How to generate two results
This project is done by **Python3**. I split codes by functions and put them into several scripts.

- Data Cleasing Script

``` python
python3 clean.py
```
This script will generate training data, stored in file *training_vectors.pkl*.

- Model 1
``` python
python3 model1_xgboost.py
```
This script will build xgboost model and do prediction, then generate the result csv file.


## Data Cleansing

## Feature Selection

## Training Data format

## Modeling & Tuning

### xgboosting

### adaboost

## Prediction



