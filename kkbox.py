import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import csv

# songs preprocess=========================

print('preprocess songs data')

# read songs
songs = pd.read_csv('songs.csv')

# pop useless attributes
songs.pop('artist_name')
songs.pop('composer')
songs.pop('lyricist')

# fillna
songs.loc[songs['genre_ids'].isnull(), 'genre_ids'] = (-10)
songs.loc[songs['language'].isnull(), 'language'] = (-1)

# genre_ids ,language get_dummies
genre = songs.pop('genre_ids')
language = songs.pop('language')

languagedummy = pd.get_dummies(language.astype(str), prefix='language_')
genredummmy = genre.str.get_dummies()
listname = []
for col in genredummmy.columns:
    listname.append('genre_' + col)
genredummmy.columns = listname

songs = pd.concat([songs, genredummmy], axis=1)
songs = pd.concat([songs, languagedummy], axis=1)

# output songs after preprocess
songs.to_csv('songsdummy.csv',
             index=False)
print('preprocess of songs data done')


# members preprocess=========================


print('preprocess members data')
# read members
members = pd.read_csv(open(
    'members.csv', newline=''))

# fillna
members.loc[members['gender'].isnull(), 'gender'] = 'unknow'

# list of cut
agelist = list(range(-5, 81, 5))

# edit outlier
members.loc[members['bd'] > 80, 'bd'] = 0

# processing each attributes
city = members.pop('city')
bd = members.pop('bd')
gender = members.pop('gender')
via = members.pop('registered_via')
init = members.pop('registration_init_time')
expiration = members.pop('expiration_date')
last = expiration - init
expiration = pd.concat([expiration, last], axis=1)
expiration.columns = ['expiration_date', 'remaining_time']

# get_dummy
city = pd.get_dummies(city.astype(str), prefix='city_')
via = pd.get_dummies(via.astype(str), prefix='via_')
gender = pd.get_dummies(gender)


# age cut
bd = pd.cut(bd, agelist)

# age get_dummy
bd = pd.get_dummies(bd, prefix='bd')


# merge attributes
members = pd.concat([members, city], axis=1)
members = pd.concat([members, bd], axis=1)
members = pd.concat([members, gender], axis=1)
members = pd.concat([members, via], axis=1)
members = pd.concat([members, init], axis=1)
members = pd.concat([members, expiration], axis=1)

# output members after preprocess
members.to_csv('membersdummy.csv',
               index=False)

print('preprocess of members data done')


# training and predicting=========================
print('preprocess training data')

# read training data
pretrain = pd.read_csv(
    'train.csv')
# read testing data
pretest = pd.read_csv(
    'test.csv')

# pop ans of training data
ans = pretrain.pop('target')

# merge train and test to normalize
total = pretrain.append(pretest)

# fillna
total.loc[total['source_system_tab'].isnull(
), 'source_system_tab'] = 'source_system_tab_none'

total.loc[total['source_screen_name'].isnull(
), 'source_screen_name'] = 'source_screen_name_none'

total.loc[total['source_type'].isnull(
), 'source_type'] = 'source_type_none'

print('preprocess of total done')

# get_dummy
total = pd.get_dummies(data=total, columns=[
                       'source_system_tab', 'source_screen_name', 'source_type'])

# read or just use variable
members = pd.read_csv(
    'membersdummy.csv', index_col='msno')

songs = pd.read_csv(
    'songsdummy.csv', index_col='song_id')

# merge total and songs,members
total.set_index('msno', inplace=True)
total = total.join(members)
total.set_index('song_id', inplace=True)
total = total.join(songs)

print('already merge songs,members')


# scalization
scaler = MinMaxScaler()
scaler.fit(total)


# sep total to train, test
train = total[:pretrain.shape[0]]
test = total[pretrain.shape[0]:]


# can modify variable
clf = linear_model.SGDClassifier(n_jobs=-1)
clf.fit(train, ans.astype('int'))
result = clf.predict(test)

# output result
result.to_csv('result.csv',
              index=False, quoting=2)
