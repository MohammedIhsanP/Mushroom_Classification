# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:01:40 2022

@author: MOHAMMED IHSAN P
"""

#import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('mushrooms.csv')
#in stalk-root column the '?' represents missing values. So we have to convert it into null values
data.replace({'?': np.nan}, inplace=True)
#here we will fill missing values of stalk-root column with mode, with respect to stalk-shape categories
data.loc[(data['stalk-root'].isna()) & (data['stalk-shape']=='t'),'stalk-root']=data[data['stalk-shape']=='t']['stalk-root'].mode()[0]
data.loc[(data['stalk-root'].isna()) & (data['stalk-shape']=='e'),'stalk-root']=data[data['stalk-shape']=='e']['stalk-root'].mode()[0]

#one hot encoding
data1=pd.get_dummies(data[['cap-shape','cap-surface','veil-type','gill-attachment']])

#concat data frames data and data1
data=pd.concat([data,data1], axis=1)
data=data.drop(['cap-shape','cap-surface','veil-type','gill-attachment'],axis=1)

#label encoding
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
for i in data[['class','cap-color','bruises','odor','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]:
    data[i]=label_en.fit_transform(data[i])
    
data=data.drop(['gill-attachment_f','gill-attachment_a','veil-type_p'],axis=1)

#split the data set into target and features
y = data['class']
x = data.drop(['class'], axis=1)

#split the data set into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)

#import libraries 
from sklearn.ensemble import RandomForestClassifier
#create the instance of the model
rf=RandomForestClassifier()
#train the data
rf.fit(x_train,y_train)
#predict x_test
y_predict=rf.predict(x_test)

#drop the features having lower feature importance.
x.drop(['stalk-color-above-ring','stalk-shape','stalk-color-below-ring','ring-number','cap-color','cap-surface_f','cap-surface_s','veil-color','cap-shape_b','cap-shape_x','cap-surface_y','cap-shape_f','cap-shape_s','cap-shape_k','cap-shape_c','cap-surface_g'], axis=1, inplace=True)

#split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42, test_size=0.3)

#import libraries 
from sklearn.ensemble import RandomForestClassifier
#create the instance of the model
rf=RandomForestClassifier()
#train the data
rf.fit(x_train,y_train)
#predict x_test
y_predict=rf.predict(x_test)

# save the model to disk

import pickle
pickle.dump(rf,open('model.pkl','wb'))