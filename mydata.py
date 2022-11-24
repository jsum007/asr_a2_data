# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dfUi_ye4gwMTKaKiW_RReNwBXPgRmnao
"""

import os
import pandas as pd

pd.__version__

pd.options.display.max_columns = None
pd.options.display.max_rows = None

data_path = 'data'

# load training log
train_df = pd.read_csv(os.path.join(data_path, 'prediction_log/train_log.csv'))

# let's look at the first 5 records
train_df.head()
# description of feature can be found here: http://moocdata.cn/data/user-activity

# let's look at the last 5 records
train_df.tail()

# read the ground truch for training data
train_truth_df = pd.read_csv(os.path.join(data_path, 'prediction_log/train_truth.csv'), index_col='enroll_id')

# 1 -> drop-out; and 0 -> not-drop-out
train_truth_df.head()

train_truth_df.tail()

# load test logs
test_df = pd.read_csv(os.path.join(data_path, 'prediction_log/test_log.csv'))
test_truth_df = pd.read_csv(os.path.join(data_path, 'prediction_log/test_truth.csv'), index_col='enroll_id')

# cobmine train and test truth
all_truth_df = pd.concat([train_truth_df, test_truth_df])

# combine train and test logs
all_log_df = pd.concat([train_df, test_df])

all_log_df.head()

all_log_df.tail()

# remove duplicate enroll_ids
train_enroll_ids = list(set(list(train_df['enroll_id'])))
test_enroll_ids = list(set(list(test_df['enroll_id'])))

# let's check total # of records on train and test datasets
print(len(train_enroll_ids))
print(len(test_enroll_ids))

# count all the actions for each user
user_action_count_df = all_log_df.groupby('enroll_id').count()[['action']]

user_action_count_df

# give columns names
user_action_count_df.columns = ['action_count']

user_action_count_df.head()

# create online session_enroll df dropping all duplicate session_ids
session_enroll_df = all_log_df[['session_id']].drop_duplicates()

session_enroll_df.head()

session_count_df = all_log_df.groupby('enroll_id').count()

session_count_df.head()

# if two columns are equal; just keep one!
user_action_count_df['action_count'].equals(session_count_df['session_id'])

session_count_df['action'].equals(session_count_df['session_id'])

# not addiding session_count as a feature becuase it's identical with action_count feature
# user_action_count_df['session_count'] = session_count['session_id']

user_action_count_df.tail()

# user action categories
video_actions = ['seek_video','play_video','pause_video','stop_video','load_video']
problem_actions = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
forum_actions = ['create_thread','create_comment','delete_thread','delete_comment']
click_actions = ['click_info','click_courseware','click_about','click_forum','click_progress']
close_actions = ['close_courseware']

for action in video_actions + problem_actions + forum_actions + click_actions + close_actions:
    action_label = action+'_count'
    action_ = (all_log_df['action'] == action).astype(int)
    all_log_df[action_label] = action_
    action_num = all_log_df.groupby('enroll_id').sum()[[action_label]]
    user_action_count_df = pd.merge(user_action_count_df, action_num, left_index=True, right_index=True)

user_action_count_df.head()

user_action_count_df.tail()

user_action_count_df.describe()

user_action_count_df.shape

user_action_count_df.describe().to_csv(os.path.join(data_path, 'features_statistics.csv'))

user_action_count_df = pd.merge(user_action_count_df, all_truth_df, left_index=True, right_index=True)

user_action_count_df.head()

user_action_count_df.tail()

# remove duplicates based on username, course_id and enroll_id
enroll_info_df = all_log_df[['username','course_id','enroll_id']].drop_duplicates()

enroll_info_df.head()

enroll_info_df.tail()

enroll_info_df.index = enroll_info_df['enroll_id']
del enroll_info_df['enroll_id']

enroll_info_df.tail()

user_action_count_df = pd.merge(user_action_count_df, enroll_info_df, left_index=True, right_index=True)

user_action_count_df.head()

# save into corresponding train and test data set
user_action_count_df.loc[test_enroll_ids].to_csv(os.path.join(data_path, 'test_features.csv'))
user_action_count_df.loc[train_enroll_ids].to_csv(os.path.join(data_path, 'train_features.csv'))

"""## Feature Preprocessing
- preprocess user and course features
"""


