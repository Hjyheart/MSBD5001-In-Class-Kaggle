# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from clean import predict_data_prpare, clean_test

# parameters for xgboost, after tuning. Check tuning process in
# notebook xgboost.ipynb
params = {
    'learning_rate': 0.1,
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 7,
    'lambda': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}


if __name__ == "__main__":
    # Load training vecotrs
    t_v = joblib.load('./training_vectors.pkl')
    X = t_v['X']
    Y = t_v['Y']

    # Build xgboost model
    plst = params.items()
    dtrain = xgb.DMatrix(X, Y)
    model = xgb.train(plst, dtrain, 500)

    # Load testing data and others
    test_df = pd.read_csv('../Data/test.csv')
    cats = joblib.load('./cats.pkl')
    genres = joblib.load('./genres.pkl')
    tags = joblib.load('./tags.pkl')

    # Create testing vectors
    test_df = clean_test(test_df)
    test_X = predict_data_prpare(test_df, cats, genres, tags)

    # Predict
    test_x = xgb.DMatrix(test_X)
    ans = model.predict(test_x)

    # Smooth the result
    for i in range(len(ans)):
        ans[i] = round(ans[i], 3)
        if ans[i] > 1:
            ans[i] = 1.0
    
    ans *= 113.8

    print(ans)

    output = pd.DataFrame({'playtime_forever': ans}, columns=['playtime_forever'])\
        .to_csv('./results/xgboost_result_t.csv', index_label='id')