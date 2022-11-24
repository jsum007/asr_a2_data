# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import pandas as pd
import numpy as np
import pickle as pkl
import math
from sklearn.preprocessing import StandardScaler

def age_convert(year):
    if year == None or math.isnan(year):
        return 0
    age = 2018 - int(year)
    if age > 70 or age < 10:
        age = 0
    return age

def gender_convert(gender):
    if gender == 'm':
        return 1
    elif gender == 'f':
        return 2
    else:
        return 0

def edu_convert(edu):
    edus = ["Bachelor's","High", "Master's", "Primary", "Middle","Associate","Doctorate"]
    #if x == None or or math.isnan(x):
    #    return 0
    if not isinstance(edu, str):
        return 0
    eduIndex = edus.index(edu)
    return eduIndex+1

def category_convert(category):
    if not isinstance(category, str):
        return 0
    categories = ['math','physics','electrical', 'computer','foreign language', 
                'business', 'economics','biology','medicine','literature','philosophy',
                'history','social science', 'art','engineering','education','environment','chemistry']
                
    catIndex = categories.index(category)
    return catIndex+1

data_path = 'data'

train_features_df = pd.read_csv(os.path.join(data_path, 'train_features.csv'), index_col=0)
test_features_df = pd.read_csv(os.path.join(data_path, 'test_features.csv'), index_col=0)

train_features_df.head()

train_features_df.tail()

all_features_df = pd.concat([train_features_df, test_features_df])

all_features_df.shape

user_profile_df = pd.read_csv(os.path.join(data_path, 'user_info.csv'), index_col='user_id')

user_profile_df.head()

user_profile_df.tail()

# extract user age
birth_year_dict = user_profile_df['birth'].to_dict()

birth_year_dict

# add user's age as feature
all_features_df['age'] = [age_convert(birth_year_dict.get(int(u), None)) for u in all_features_df['username']]

all_features_df.head()

# extract user gender
user_gender = user_profile_df['gender'].to_dict()
# add gender feature to all_features
all_features_df['gender'] = [gender_convert(user_gender.get(int(u), None)) for u in all_features_df['username']]

all_features_df.head()

# extract education and add it as a feature
user_edu = user_profile_df['education'].to_dict()
all_features_df['education'] = [edu_convert(user_edu.get(int(u), None)) for u in all_features_df['username']]

user_enroll_count_df = all_features_df.groupby('username').count()[['course_id']]

user_enroll_count_df

user_enroll_count_df.columns = ['user_enroll_count']

course_enroll_count_df = all_features_df.groupby('course_id').count()[['username']]

course_enroll_count_df.head()

course_enroll_count_df.columns = ['course_enroll_count']

all_features_df = pd.merge(all_features_df, user_enroll_count_df, left_on='username', right_index=True)

all_features_df.head()

all_features_df = pd.merge(all_features_df, course_enroll_count_df, left_on='course_id', right_index=True)

all_features_df.head()

all_features_df.tail()

#extract course category
course_info_df = pd.read_csv(os.path.join(data_path, 'course_info.csv'), index_col='id')
category_dict = course_info_df['category'].to_dict()

category_dict

# add course_category feature
all_features_df['course_category'] = [category_convert(category_dict.get(str(x), None)) for x in all_features_df['course_id']]

all_features_df.head()

# remove id features
bad_features = ['username', 'course_id']# let's clean drop some features just to see...
all_features_df.drop(columns=bad_features, inplace=True)

all_features_df.head()

all_features_df.shape

numeric_features = [c for c in all_features_df.columns if 'count' in c or 'time' in c or 'num' in c]

numeric_features

len(numeric_features)

# dump actual features as a pickle file
pkl.dump(numeric_features, open(os.path.join(data_path, 'act_features.pkl'), 'wb'))

all_feature_names = numeric_features + ['age']

# perform Standard Scalar Transformation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
scaled_df = scaler.fit_transform(all_features_df[numeric_features])

scaled_df.shape

scaled_df

for i, n_f in enumerate(numeric_features):
    print(i, n_f)
    all_features_df[n_f] = scaled_df[:,i]

all_features_df.head()

all_features_df.tail()

#extract user cluster
#cluster_label = np.load('cluster/label_5_10time.npy', allow_pickle=True)
#user_cluster_id = pkl.load(open('cluster/user_dict','r'))

#all_feat['cluster_label'] = [cluster_label[user_cluster_id[u]] for u in all_feat['username']]

# save training and testing as CSV files
all_features_df.loc[train_features_df.index].to_csv(os.path.join(data_path, 'train_normalized_features.csv'))
all_features_df.loc[test_features_df.index].to_csv(os.path.join(data_path, 'test_normalized_features.csv'))

# save single file with all features
all_features_df.to_csv(os.path.join(data_path, 'all_normalized_features.csv'))
