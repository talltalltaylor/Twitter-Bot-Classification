import framer
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import cm
import matplotlib.pyplot as plt

# instantiate new Framer
f = framer.Framer()
"""
# get data from csv for bots and humans
bot_frame = f.get_from_csv('bots.1k.csv')
bf = f.make_labels(bot_frame, bot=True)
human_frame = f.get_from_csv('humans.1k.csv')
hf = f.make_labels(human_frame)
"""
bot_names = ['help_laura', 'GWhV8mU5pMy2Ier', 'AnthonyBurnha19', 'IgnotoLugar', 'Lady_Sybilla', 'AllMoviesCorp',
             'batianhu56', 'IvyFoliage', 'Zeus24513824', 'mohanLa19052950', 'TrooHefner', 'P48063445',
             'carly04633486', 'Vanessa79764758', 'bold_picture', 'thinkpraybot', 'vtmtst', '1104Bj',
             'Lxxxx3326', 'Inmycrescentph1', 'quivadaq']
real_bots = f.get_from_json('period3.json', bots=bot_names)
rb = f.make_labels(real_bots, bot=True)
real_humans = f.get_from_json('period3.json')
real_humans = real_humans.sample(frac=0.0007)
rh = f.make_labels(real_humans)

# combine data and shuffle to represent real world example of fake accounts
data = f.human_bot_sim(rh, rb)

# make final features/labels frames
features = pd.DataFrame(data, columns=['Favorites per Tweet', 'Follow Ratio', 'Profile Age',
                                       'Retweets per Tweet', 'Tweet Count', 'User Favorites', 'User Lists'])
labels = pd.DataFrame(data, columns=['bot'])

# convert labels
labels = np.ravel(labels)
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

# stratified cross validation to try and preserve the percentage of labels in the samples/subsets
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=0)

for train_index, test_index in sss.split(features, labels):
    train, test = features.iloc[train_index], features.iloc[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]


gnb = GaussianNB()
lr = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=10)


def crystal_ball(model):
    model.fit(train, train_labels)
    predictions = model.predict(test)
    print(accuracy_score(test_labels, predictions))

    return predictions


pred1 = crystal_ball(gnb)
pred2 = crystal_ball(lr)
pred3 = crystal_ball(dt)
pred4 = crystal_ball(rf)




