#libarires needed
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pickle

#load dataset
dataset = pd.read_csv("C:/Users/Nandini/project1(credit card default detection)/data/UCI_Credit_Card.csv")
#C:\Users\Nandini\project1(credit card default detection)\data
dataset.head()
#datafile exploration
print(dataset.shape)
dataset.info()
dataset.describe()
#EDA
dataset.isnull().sum()
features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>=1]
dataset['EDUCATION'].replace([0, 6], 5, inplace=True)
dataset.EDUCATION.value_counts()
#in the dataset there is three category of marriage but here there is 0 as well, so we can combine 0 with 3.
dataset['MARRIAGE'].replace(0, 3, inplace=True)
dataset=dataset.drop(["ID"], axis=1)
dataset=dataset.rename(columns={'default.payment.next.month':'defaultpayment'})

#data analysis
dataset.defaultpayment.value_counts()
df = dataset.loc[:, ['AGE', 'EDUCATION', 'SEX', 'defaultpayment']]
#to make bar graph
df.iloc[1:20].plot(kind = 'bar',figsize = (20,10))
#to make histogram
df.hist(bins=20, figsize=(15,8),layout=(1,4))
plt.show()
age_default = df.groupby('AGE')['defaultpayment'].mean()
plt.scatter(age_default.index, age_default.values)
plt.xlabel('Age')
plt.ylabel('Default Payment')
plt.title('Relationship between Age and Default Payment')
plt.show()
#to check if there is any correlation between default payment and age
corr_matrix = df.corr()
print(corr_matrix['defaultpayment']['AGE'])
corr_matrix = df.corr()
print(corr_matrix['defaultpayment']['SEX'])
# Variable correlation
plt.figure(figsize=(10,8))
sns.set(font_scale=1)
sns.heatmap(df.corr(), vmax=1, square=True, annot=True,cmap='viridis')
plt.title('Correlation between different attributes')
plt.show()
df.hist(figsize=(20,10),color = 'r' , alpha = .9)
#Pairplot
plt.figure(figsize=(20,30*10))

features = dataset.iloc[:,6:25].columns
gs = gridspec.GridSpec(30, 1)
for i, feature in enumerate(dataset[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(dataset[feature][dataset.defaultpayment == 1], bins=50)
    sns.distplot(dataset[feature][dataset.defaultpayment == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('Feature: ' + str(feature))
plt.show()

warnings.filterwarnings('ignore')
features = dataset.iloc[:,6:25].columns
# Pie chart for the default payment variable distribution
defaulty =len(dataset[dataset['defaultpayment']==1])
notdefaulty = len(dataset[dataset['defaultpayment']==0])

# Data to plot
labels = 'default','Not default'
sizes = [defaulty,notdefaulty]

# Plot
plt.figure(figsize=(5,8))
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=0)
plt.title('Ratio of defaulty vs not defaulty\n', fontsize=20)
sns.set_context("paper", font_scale=1)

plt.show()

#model preperation
#dependent and independent variables
x = dataset.drop('defaultpayment',axis=1)
y = dataset['defaultpayment']
# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
# concatenate our training data back together
x = pd.concat([x_train, y_train], axis=1)
x.head()
# Upsampling to balance the default and not default ratio
not_default = x[x.defaultpayment==0]
default = x[x.defaultpayment==1]

# upsample minority
default_upsampled = resample(default,
                          replace=True,
                          n_samples=len(not_default),
                          random_state=101)

# combine majority and upsampled minority
upsampled = pd.concat([not_default, default_upsampled])

# check new class counts
upsampled.defaultpayment.value_counts()
#1.logistic regression
y_train = upsampled.defaultpayment
x_train = upsampled.drop('defaultpayment', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(x_train, y_train)

upsampled_pred = upsampled.predict(x_test)
# Accuracy score
accuracy_score(y_test, upsampled_pred)
# Confusion Matrix
pd.DataFrame(confusion_matrix(y_test, upsampled_pred))
print('Logistic Regression classification_report')

print('...'*10)

print(classification_report(y_test,upsampled_pred))
# Upsampling to balance the default and not default ratio
not_default = x[x.defaultpayment==0]
default = x[x.defaultpayment==1]

# upsample minority
default_upsampled = resample(default,
                          replace=True,
                          n_samples=int(0.6*len(not_default)),
                          random_state=101)

# combine majority and upsampled minority
upsampled= pd.concat([not_default, default_upsampled])

# check new class counts
upsampled.defaultpayment.value_counts()
#0.6, which means that the number of samples in the minority class after oversampling will be 60% of the number of samples in the majority class.
#2.logistic regression with unequal upsampled data
y_train = upsampled.defaultpayment
x_train = upsampled.drop('defaultpayment', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(x_train, y_train)

upsampled_pred = upsampled.predict(x_test)
accuracy_score(y_test, upsampled_pred)
print('Logistic Regression classification_report on DownSampling')

print('...'*10)

print(classification_report(y_test,upsampled_pred))
#so it is seen that the accuracy is higher when the data is divided in 60-40 ration in case of logistic regression.
#random forest
# Running the random forest with default parameters.
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
#training the model on training data
rfc.fit(x_train, y_train)
#predict the target variable for test data
randf_pred = rfc.predict(x_test)
accuracy_score(y_test, randf_pred)
print(f"The model prediction on train dataset: {round(rfc.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(rfc.score(x_test, y_test),2)}")

#naives bayes
gnb = GaussianNB()
gnb_best = gnb.fit(x_train, y_train)
gnb_pred = gnb_best.predict(x_test)
accuracy_score(y_test, gnb_pred)

gnb = BernoulliNB()
gnb_best = gnb.fit(x_train, y_train)
gnb_pred = gnb_best.predict(x_test)
accuracy_score(y_test, gnb_pred)

#SVC with gridsearch CV
# Running 5 fold crossvalidation
C = [1]
gammas = [0.001, 0.1]
param_grid = dict(C=C, gamma=gammas)

svm = SVC(kernel='rbf', probability=True)
svm_grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1)
svm_grid.fit(x_train, y_train)

# predict on test set
grid_pred = svm_grid.predict(x_test)
accuracy_score(y_test, grid_pred)

#decision tree
# decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(x_train, y_train)

# Make predictions on test data
clf_pred = clf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, clf_pred)
print("Accuracy: {:.2f}".format(accuracy))

#adaboost

#AdaBoost classifier with Decision Tree as base estimator
ada = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the model
ada.fit(x_train, y_train)

# Make predictions on test data
ada_pred = ada.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, ada_pred)
print("Accuracy: {:.2f}".format(accuracy))

#gradient boosting
#Gradient Boosting classifier
gra = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gra.fit(x_train, y_train)

# Make predictions on test data
gra_pred = gra.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, gra_pred)
print("Accuracy: {:.2f}".format(accuracy))
print(f"The model prediction on train dataset: {round(gra.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(gra.score(x_test, y_test),2)}")

#BEST MODEL SAVE
pickle.dump(rfc, open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))


