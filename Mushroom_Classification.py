#!/usr/bin/env python
# coding: utf-8

# # Competition

# ### Submitted by: Mohammed Ihsan P, Aleena Francis, Rustham Shahan V, Alka Sherine Benny, Reuben M Sunil

# ## Domain-Agriculture

# Mushroom Classification-:
# 
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.

# ## Dataset Description:
# Attribute Information: (classes: edible=e, poisonous=p)
# 
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 
# bruises: bruises=t,no=f
# 
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# 
# gill-attachment: attached=a,descending=d,free=f,notched=n
# 
# gill-spacing: close=c,crowded=w,distant=d
# 
# gill-size: broad=b,narrow=n
# 
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 
# stalk-shape: enlarging=e,tapering=t
# 
# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# 
# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# veil-type: partial=p,universal=u
# 
# veil-color: brown=n,orange=o,white=w,yellow=y
# 
# ring-number: none=n,one=o,two=t
# 
# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 
# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 
# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 
# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# In[1]:


#import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('mushrooms.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# # --PreProcessing

# ## 1)- Handle missing values-:

# In[5]:


#check values in stalk-root column
data['stalk-root'].value_counts()


# In[6]:


#in stalk-root column the '?' represents missing values. So we have to convert it into null values
data.replace({'?': np.nan}, inplace=True)


# In[7]:


data['stalk-root'].value_counts()


# In[8]:


#check for missing values
data.isna().sum()


# In[9]:


data.dtypes


# In[10]:


data['stalk-shape'].value_counts()


# In[11]:


data['stalk-root'].value_counts()


# In[12]:


#here we will fill missing values of stalk-root column with mode, with respect to stalk-shape categories
data.loc[(data['stalk-root'].isna()) & (data['stalk-shape']=='t'),'stalk-root']=data[data['stalk-shape']=='t']['stalk-root'].mode()[0]
data.loc[(data['stalk-root'].isna()) & (data['stalk-shape']=='e'),'stalk-root']=data[data['stalk-shape']=='e']['stalk-root'].mode()[0]


# In[13]:


data.isna().sum()


# ## 2)-Encoding-:

# In[14]:


#one hot encoding
data1=pd.get_dummies(data[['cap-shape','cap-surface','veil-type','gill-attachment']])

#concat data frames data and data1
data=pd.concat([data,data1], axis=1)
data=data.drop(['cap-shape','cap-surface','veil-type','gill-attachment'],axis=1)


# In[15]:


data.head()


# In[16]:


#label encoding
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
for i in data[['class','cap-color','bruises','odor','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]:
    data[i]=label_en.fit_transform(data[i])


# In[17]:


data.head()


# In[18]:


data.shape


# ## 3)-Feature Reduction-:

# In[19]:


data.columns


# In[20]:


#heatmap
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True , cmap='YlGnBu')
plt.tight_layout()
plt.show()


# gill attachment_f and gill attachment_a has strong correlation with veil color (one is strong positive and other is strong negative). So when we check the correlation between these columns and target column 'class', veil color has more correlation in both cases. Hence we could drop the other two columns.
# Since veil type_P has only one value for all the rows, we could drop this also.

# In[21]:


data=data.drop(['gill-attachment_f','gill-attachment_a','veil-type_p'],axis=1)


# In[22]:


data.head()


# # --Exploratory Data Analysis

# In[23]:


data.head()


# In[24]:


data.tail()


# In[25]:


data.info()


# In[26]:


data.shape


# In[27]:


data.columns


# In[28]:


data.describe()


# In[29]:


#pair plot
slice_data=data[['class','cap-color', 'bruises', 'gill-size','stalk-root','population', 'habitat']]
plt.figure(figsize=(10,10))
sns.pairplot(slice_data)
plt.tight_layout()
plt.show()


# In[30]:


#scatter plot
plt.figure(figsize=(5,3))
sns.scatterplot(x=data['ring-type'],y=data['bruises'])
plt.title('Scatterplot between Ring type and bruises')
plt.xlabel('ring-type')
plt.tight_layout()
plt.show()


# In[31]:


#scatter plot
plt.figure(figsize=(7,5))
sns.scatterplot(x=data['population'],y=data['habitat'], hue=data['class'])
plt.title('Scatterplot between population and habitat with respect to class')
plt.xlabel('population')
plt.tight_layout()
plt.show()


# In[32]:


#violin plot
sns.violinplot(data['population'])
plt.title('Violin Plot of Population')
plt.show()


# In[33]:


#boxplot
sns.boxplot(data['habitat'])
plt.title('Box Plot of Habitat')
plt.show()


# In[34]:


#countplot
sns.countplot(data['stalk-shape'])
plt.title('Countplot of Stalk-shape')
plt.show()


# Here 0 corresponds to enlarging shape and 1 corresponds to tapering shape.

# In[35]:


#heatmap
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True , cmap='YlGnBu')
plt.tight_layout()
plt.show()


# Insights-:
#     
# * There is high positive correlation between ring-type and bruises.
# 
# * There is high negative correlation between cap-shape_f and cap-shape_x.
# 
# * There is high positive correlation between ring-type and gill color.
# 
# * There is high negative correlation between gill-spacing and population.
# 
# * There is high positive correlation between spore-print-color and gill-size.

# # --Modelling

# ## Split the dataset

# In[36]:


#split the data set into target and features
y = data['class']
x = data.drop(['class'], axis=1)


# In[37]:


#split the data set into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)


# In[38]:


#check how many data points are there in the training set
x_train.shape


# ## 1-Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model = logit_model.fit(x_train, y_train)
y_predict = logit_model.predict(x_test)


# In[40]:


#check the performance of the model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[41]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# In[42]:


#check confusion matrix
confusion_matrix(y_test, y_predict)


# There are 98 misclassifications, we have to deal with.

# ## 2)-kNN

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
#create model with varied k values
acc_values = []
#take 3 to 15 random values for k
neighbors = np.arange(3,15)
#loop to create kNN model for each k values
for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    acc_values.append(acc)


# In[44]:


print(acc_values)


# In[45]:


plt.plot(neighbors, acc_values, 'o-')
plt.xlabel('k-values')
plt.ylabel('Accuracy')
plt.show()


# We will take k value as 6

# In[46]:


#replace k as 6
classifier = KNeighborsClassifier(n_neighbors=6, metric='minkowski')
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)


# In[47]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# ## 3)-SVM

# ### Linear SVM

# In[48]:


#import library
from sklearn.svm import SVC
#create an instance of the model
svm_linear = SVC(kernel='linear')
svm_linear.fit(x_train, y_train)
y_predict = svm_linear.predict(x_test)


# In[49]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# ### Radial Based Function SVM

# In[50]:


#import library
from sklearn.svm import SVC
#create an instance of the model
svm_radial = SVC(kernel='rbf')
svm_radial.fit(x_train, y_train)
y_predict = svm_radial.predict(x_test)


# In[51]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# In[52]:


#check confusion matrix
confusion_matrix(y_test, y_predict)


# There are 33 misclassifications we have to deal with.

# ## 4)-Desicion Tree Classifier

# In[53]:


#import library
from sklearn.tree import DecisionTreeClassifier
#create an instance of the model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
y_predict = dt_model.predict(x_test)


# In[54]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# ## 5)-Random Forest Classifier

# In[55]:


#import libraries 
from sklearn.ensemble import RandomForestClassifier
#create the instance of the model
rf=RandomForestClassifier()
#train the data
rf.fit(x_train,y_train)
#predict x_test
y_predict=rf.predict(x_test)


# In[56]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# ## 6)-Gradient Boosting Classifier

# In[57]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
#predict the y
y_predict=gb.predict(x_test)


# In[58]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# # --Model Fine Tuning

# Among the above models, kNN, Decision Tree and Random forest models have 100% accuracy, so there is no need of fine tuning. Hence we will fine tune the remaining models.

# ## Logistic Regression

# In[59]:


from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression(penalty='l2' ,C=5, solver='sag', max_iter=200, multi_class='auto', verbose=0, warm_start=True, n_jobs=None)
logit_model = logit_model.fit(x_train, y_train)
y_predict = logit_model.predict(x_test)


# In[60]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# By changing the hyper parameters of the model, we increased the performance of the model

# ## SVM

# ### Linear SVM

# In[61]:


#create an instance of the model
svm_linear = SVC(kernel='linear', C=2.0, max_iter=- 1 )
svm_linear.fit(x_train, y_train)
y_predict = svm_linear.predict(x_test)


# In[62]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# By changing the hyper parameters of the model, we increased the performance of the model

# ### Radial Based Function SVM

# In[63]:


#create an instance of the model
svm_radial = SVC(kernel='rbf',C=2.0, max_iter=- 1, gamma='auto')
svm_radial.fit(x_train, y_train)
y_predict = svm_radial.predict(x_test)


# In[64]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# By changing the hyper parameters of the model, we increased the performance of the model

# ## Gradient Boosting Classifier

# In[65]:


gb=GradientBoostingClassifier(n_estimators=150, subsample=1, criterion='friedman_mse', min_samples_split=3)
gb.fit(x_train,y_train)
#predict the y
y_predict=gb.predict(x_test)


# In[66]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# By changing the hyper parameters of the model, we increased the performance of the model

# In[ ]:





# # Feature importance-

# In[67]:


pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)*100


# In[68]:


features_list = x.columns.values
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8,7))


plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
#plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')
plt.show()


# In[69]:


#drop the features having lower feature importance.
x.drop(['stalk-color-above-ring','stalk-shape','stalk-color-below-ring','ring-number','cap-color','cap-surface_f','cap-surface_s','veil-color','cap-shape_b','cap-shape_x','cap-surface_y','cap-shape_f','cap-shape_s','cap-shape_k','cap-shape_c','cap-surface_g'], axis=1, inplace=True)


# In[70]:


#split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42, test_size=0.3)


# In[71]:


#import libraries 
from sklearn.ensemble import RandomForestClassifier
#create the instance of the model
rf=RandomForestClassifier()
#train the data
rf.fit(x_train,y_train)
#predict x_test
y_predict=rf.predict(x_test)


# In[72]:


#check the performance of the model
print('Accuracy is :', accuracy_score(y_test,y_predict))
print('Precision is :', precision_score(y_test,y_predict))
print('Recall is :', recall_score(y_test,y_predict))
print('f1 Score is :', f1_score(y_test,y_predict))


# In[74]:


# save the model to disk

import pickle
pickle.dump(rf,open('model.pkl','wb'))


# In[ ]:




