# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

def training_data_prpare(data, cats, genres, tags):
    """
    Prepare training vectors, the output including genres, categories, 
    tags one-hot and is_free, time_gap and average_review

    Param:
        data: dataframe
        cats, genres, tags: feature list
    Output:
        X, Y
    """
    X = []
    Y = []
    for item in data.iterrows():
        row = item[1]
        tmp = []
        Y.append(row['playtime_forever']/113.8)
        # genres
        row_g = row['genres'].split(',')
        for g in genres:
            if g in row_g:
                tmp.append(1)
            else:
                tmp.append(0)
        # cats
        row_c = row['categories'].split(',')
        for c in cats:
            if c in row_c:
                tmp.append(1)
            else:
                tmp.append(0)
        # tags
        row_t = row['tags'].split(',')
        for t in tags:
            if t in row_t:
                tmp.append(1)
            else:
                tmp.append(0)
        # is_free
        if row['is_free']:
            tmp.append(1)
        else:
            tmp.append(0)
        # price
        tmp.append(row['price'])
        # timegap
        tmp.append(row['time_gap'])
        # average_review
        tmp.append(row['average_review'])
        
        X.append(tmp)
    
    return X, Y

def predict_data_prpare(data, cats, genres, tags):
    """
    Prepare testing vectors, the output including genres, categories, 
    tags one-hot and is_free, time_gap and average_review same as training data

    Param:
        data: dataframe
        cats, genres, tags: feature list
    Output:
        X
    """
    X = []
    for item in data.iterrows():
        row = item[1]
        tmp = []
        # genres
        row_g = row['genres'].split(',')
        for g in genres:
            if g in row_g:
                tmp.append(1)
            else:
                tmp.append(0)
        # cats
        row_c = row['categories'].split(',')
        for c in cats:
            if c in row_c:
                tmp.append(1)
            else:
                tmp.append(0)
        # tags
        row_t = row['tags'].split(',')
        for t in tags:
            if t in row_t:
                tmp.append(1)
            else:
                tmp.append(0)
        # is_free
        if row['is_free']:
            tmp.append(1)
        else:
            tmp.append(0)
        # price
        tmp.append(row['price'])
        # timegap
        if not np.isnan(row['time_gap']):
            tmp.append(row['time_gap'])
        else:
            tmp.append(0.5)
        # average_review
        if not np.isnan(row['average_review']):
            tmp.append(row['average_review'])
        else:
            tmp.append(0.5)
        
        X.append(tmp)
    
    return X

def clean_test(test_df):
    """
    Clean testing data and create new features

    Param:
        test_df: testing dataframe
    Output:
        test_df after normalization and add new features
    """
    test_df['average_review'] = test_df['total_positive_reviews'] / (test_df['total_positive_reviews'] + test_df['total_negative_reviews'])
    test_df['purchase_date'] = pd.to_datetime(test_df['purchase_date'], infer_datetime_format=True)
    test_df['release_date'] = pd.to_datetime(test_df['release_date'], infer_datetime_format=True)
    test_df['time_gap'] = test_df['purchase_date'] - test_df['release_date']
    test_df['time_gap'] = (test_df['time_gap'] - timedelta(-481)) / (timedelta(4320) - timedelta(-481))
    test_df['price'] = test_df['price'] / 15999900.0

    return test_df


if __name__ == "__main__":
    # Load Data, drop NaN data
    df = pd.read_csv('../Data/train.csv')
    df.drop([5, 76], inplace=True)

    # Clean categories and store
    all_cat = {}
    for item in df.iterrows():
        for attr in item[1]['categories'].split(','):
            if attr not in all_cat.keys():
                all_cat[attr] = 1
            else:
                all_cat[attr] += 1

    all_cat = sorted(all_cat.items(), key=lambda x:x[1], reverse=True)[5:]
    cats = [x[0] for x in all_cat]

    joblib.dump(cats, 'cats.pkl')

    # Clean genres and store
    all_genres = {}
    for item in df.iterrows():
        for attr in item[1]['genres'].split(','):
            if attr not in all_genres.keys():
                all_genres[attr] = 1
            else:
                all_genres[attr] += 1
    genres = [x for x in all_genres.keys()]

    joblib.dump(genres, './genres.pkl')

    # Clean tags and store
    all_tags = {}
    for item in df.iterrows():
        for attr in item[1]['tags'].split(','):
            if attr not in all_tags.keys():
                all_tags[attr] = 1
            else:
                all_tags[attr] += 1

    # Drop high frequent tags and low frequent tags
    all_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[4:70]
    tags = [x[0] for x in all_tags]

    joblib.dump(tags, './tags.pkl')

    # Create new feature time_gap: time between purchase_data and release_data
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], infer_datetime_format=True)
    df['release_date'] = pd.to_datetime(df['release_date'], infer_datetime_format=True)
    df['time_gap'] = df['purchase_date'] - df['release_date']
    df['time_gap'] = (df['time_gap'] - df['time_gap'].min()) / (df['time_gap'].max() - df['time_gap'].min())

    # Normalization
    df['average_review'] = df['total_positive_reviews'] / (df['total_positive_reviews'] + df['total_negative_reviews'])
    df.fillna(0.0, inplace=True)
    df['price'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())

    # Create training vectors and store
    tx, ty = training_data_prpare(df, cats, genres, tags)
    joblib.dump({'X': tx, 'Y': ty}, './training_vectors.pkl')