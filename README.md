# MSBD5001 Individual Project

Thanks to professors and TA for arranging this individual project. In this project, we are supposed to predict the gaming hours according to the given data. Shortage of training data is the major problem of this project and the labels are very sparse, most of them are closed to zero.

I pick 2 models to do prediction since there are 2 results can be used in kaggle. The first one is xgboosting, a tree based model, quite popular among Kaggle projects. The second one is adaboosting, a bagging model.

I'm perssimistic about the final result due to the shortage of training data, the models are easily to get overfitting. But I think the final result isn't everthing, the key point of this project should be applying the standard process of modeling and enjoying data analysis.

Based on what I have learnt from MSBD5001, here are major steps for this project:

- Data Cleansing
- Feature Engineering
- Training Data Preparation
- Modeling & Tuning

## How to generate two results
This project is done by **Python3**. I split codes by functions and put them into several scripts.

### Packages Needed

- Pandas
- Numpy
- Sklearn
- Xgboost
- Joblib

### Commands

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

- Model 2
``` python
python3 model2_ridge_regression.py
```
This script will build ridge regression model and do prediction, then generate the result scv file.

## Data Cleansing

### nan values
Firstly, I checked nan values in every column and found two rows, row 5 and row 76 had nan values for column *total_positive_values* and *total_negative_values*. Therefore, I wiped them out.

### text-based values
Then, the most difficult part is text-based columns: genres, categories and tags. I split them by commas.

### date
There are two columns about date. In order to utilize them better, I transformed them from text to python datetime object.

## Feature Engineering
Feature engineering is the most important part of data modeling, good features are the key to success. In this project, some features are obvious and intuitive, like the price, tag and comment of games. However, some maybe the noise I don't know, so I use tree based mothod to draw the feature importance, then pick up most important ones.

### new features
Firstly, I created 2 new features from current features: **time_gap** and **average_review**.

- **time_gap**: *purchase_date* - *release_date*
    - I think the time gap between these two dates can be very useful. If you buy a game which is released just a few days, most likely you will play it more than others.
- **average_review**: *total_positive_reviews* / (*total_positive_reviews* + *total_negative_reviews*)
    - This feature is just a metric in statistic.

### feature preparation

#### genres
There are 20 different genres in total in training data. They are: 
```
['Adventure', 'Casual', 'Indie', 'RPG', 'Action', 'Strategy', 'Simulation', 'Racing', 'Sports', 'Massively Multiplayer', 'Sexual Content', 'Violent', 'Free to Play', 'Early Access', 'Audio Production', 'Gore', 'Design & Illustration', 'Nudity', 'Animation & Modeling', 'Utilities']
```

#### categories
There are 29 different categories in total in training data. They are:
```
['Single-player', 'Steam Trading Cards', 'Steam Cloud', 'Partial Controller Support', 'Full controller support', 'Multi-player', 'Steam Achievements', 'Steam Workshop', 'Co-op', 'Steam Leaderboards', 'Online Co-op', 'Local Co-op', 'Shared/Split Screen', 'Stats', 'Online Multi-Player', 'Cross-Platform Multiplayer', 'SteamVR Collectibles', 'Local Multi-Player', 'Remote Play on Phone', 'Remote Play on Tablet', 'Remote Play on TV', 'Valve Anti-Cheat enabled', 'Commentary available', 'Captions available', 'Includes level editor', 'In-App Purchases', 'VR Support', 'MMO', 'Includes Source SDK']
```

#### tags
There are 312 different tags in total in training data. 312 is too large, I don't want the final training vectors have high dimentions. So I analyzed the frequency of each tag, then filtered out tags with too high or too low frequency. Last, only 66 left, they are:
```
['Atmospheric', 'Great Soundtrack', 'Story Rich', 'Multiplayer', 'RPG', 'Open World', 'Strategy', 'Co-op', 'Fantasy', 'Sci-fi', 'Masterpiece', 'Simulation', '2D', 'First-Person', 'Puzzle', 'Third Person', 'Shooter', 'Funny', 'Casual', 'Survival', 'Difficult', 'Horror', 'Sandbox', 'Exploration', 'Female Protagonist', 'Early Access', 'Comedy', 'FPS', 'Gore', 'Point & Click', 'Online Co-Op', 'Choices Matter', 'Classic', 'Space', 'VR', 'Violent', 'Turn-Based', 'Platformer', 'Dark', 'Moddable', 'Hack and Slash', 'Local Co-Op', 'Mystery', 'Stealth', 'Nudity', 'Retro', 'Action RPG', 'Psychological Horror', 'Building', 'Isometric', 'Third-Person Shooter', 'Replay Value', 'Pixel Graphics', 'Mature', 'Walking Simulator', 'Short', 'Post-apocalyptic', 'Character Customization', 'Tactical', 'Free to Play', 'Massively Multiplayer', 'Crafting', 'Controller', 'Rogue-like', 'RTS', 'Historical']
```

### feature selection 
I put all features together and run xgboost, then drew feature importance:

![img](./Scripts/all_features.png)

Then I filtered out some features with low importance.

## Training Data format

|genres|categories|tags|is_free|price|time_gap|average_review|
|-|-|-|-|-|-|-|
|one hot|one hot|one hot|boolean|normalized|normalized|normalized|

## Modeling

### xgboosting
Parameters after tunning:

|learning_rate|booster|objective|gamma|max_depth|lambda|subsample|colsample_bytree|min_child_weight|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0.1|gbtree|reg:gamma|0.1|7|3|0.9|0.9|3|

### ridge regression

Optimized alpha: 133.4


