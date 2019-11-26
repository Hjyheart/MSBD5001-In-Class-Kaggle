# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from clean import predict_data_prpare, clean_test
from sklearn import linear_model

if __name__ == "__main__":
    # Load training vecotrs
    t_v = joblib.load('../Features/training_vectors.pkl')
    X = t_v['X']
    Y = t_v['Y']

    # Build ridge model by using tunned alpha
    rid = linear_model.Ridge(alpha=0.5)
    rid.fit(X, Y)

    # Load testing data and others
    test_df = pd.read_csv('../Data/test.csv')
    cats = joblib.load('../Features/cats.pkl')
    genres = joblib.load('../Features/genres.pkl')
    tags = joblib.load('../Features/tags.pkl')
    test_df = clean_test(test_df)
    test_X = predict_data_prpare(test_df, cats, genres, tags)

    # Predict
    ans = rid.predict(test_X)
    for i in range(len(ans)):
        if ans[i] < 0:
            ans[i] = 0
        elif ans[i] > 1:
            ans[i] = 1
        else:
            ans[i] = round(ans[i], 3)
    ans *= 113.8

    print(ans)

    output = pd.DataFrame({'playtime_forever': ans}, columns=['playtime_forever'])\
        .to_csv('../Results/ridge_result_t.csv', index_label='id')