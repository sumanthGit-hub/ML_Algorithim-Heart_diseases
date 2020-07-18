# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:23:11 2020

@author: sumanth
"""

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd

heart=pd.read_excel('heart.xlsx')
heart.shape
heart.head(10)
heart.describe()
list(heart)

heart.isnull().sum()

heart.DeathCause.describe()
heart.AgeCHDdiag.describe()
heart.AgeAtDeath.describe()

heart.Chol_Status.describe()
heart.Smoking_Status.describe()

heart.Chol_Status=heart.Chol_Status.fillna(heart.Chol_Status.mode()[0],inplace=False)
heart.Smoking_Status=heart.Smoking_Status.fillna(heart.Smoking_Status.mode()[0],inplace=False)

heart.Smoking_Status=heart.Smoking_Status.map({'Very Heavy (> 25)':'Heavy','Heavy (16-25)':'Heavy','Moderate (6-15)':'Moderate','Light (1-5)':'Light','Non-smoker':'Non-smoker'})

# DATA VISHUVALIZATION
heart['AgeAtStart'].plot.box()
heart['Height'].plot.box()
heart['Weight'].plot.box()
heart['Diastolic'].plot.box()
heart['Systolic'].plot.box()
heart['MRW'].plot.box()
heart['Smoking'].plot.box()
heart['Cholesterol'].plot.box()


heart['Status'].value_counts()
heart['Status'].value_counts(normalize=True)
heart['Status'].value_counts().plot.bar()
heart['Status'].value_counts(normalize=True).plot.bar(title='Status')


heart['Chol_Status'].value_counts().plot.bar()
heart['Smoking_Status'].value_counts().plot.bar()
heart['Sex'].value_counts().plot.bar()
heart['BP_Status'].value_counts().plot.bar()

sns.distplot(heart['AgeAtStart'])
sns.distplot(heart['Height'])
sns.distplot(heart['Weight'])
sns.distplot(heart['Diastolic'])
sns.distplot(heart['Systolic'])
sns.distplot(heart['MRW'])
sns.distplot(heart['Smoking'])
sns.distplot(heart['Cholesterol'])


H1=pd.crosstab(heart['Status'],heart['Chol_Status'])
H2=pd.crosstab(heart['Status'],heart['Smoking_Status'])
H3=pd.crosstab(heart['Status'],heart['Sex'])
H4=pd.crosstab(heart['Status'],heart['BP_Status'])

## target variable and x variables

H1.div(H1.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
H2.div(H2.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
H3.div(H3.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
H4.div(H4.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)






X=heart.iloc[:,4:]
X1=X.drop(X[['AgeAtDeath']],axis=1)
list(X1)
X1.head()
Y=heart.Status

c=X1.corr()
sns.heatmap(c,annot=True)

X2=pd.get_dummies(X1)
#heart['Status'].replace('Alive',0,inplace=True)
#heart['Status'].replace('Dead',1,inplace=True)
Y2=heart.Status.map({'Alive':0,'Dead':1})
list(X2)

from sklearn import linear_model
lm=linear_model.LogisticRegression().fit(X2,Y2)
a=lm.score(X2,Y2)
pred=lm.predict(X2)

from sklearn import metrics
ACC=metrics.accuracy_score(pred,Y2)

from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X2,Y2,stratify=Y2,random_state=20)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


lm=linear_model.LogisticRegression().fit(X_train,Y_train)
pred_y=lm.predict(X_test)

from sklearn import metrics
Acc=metrics.accuracy_score(pred_y,Y_test)


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(Y_test, pred_y)
auc = metrics.roc_auc_score(Y_test, pred_y)
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label='Validation,auc='+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
plt.close()

###########################################################################

from sklearn import model_selection
model=model_selection.StratifiedKFold(n_splits=6,random_state=5,shuffle=False)

i=1
for train_index ,test_index in model.split(X2,Y2):
    print('\n{} of fold {}'.format(i, model.n_splits))
    xtr, xte = X2.loc[train_index],X2.loc[test_index]
    ytr, yte = Y2.loc[train_index],Y2.loc[test_index]

    #xtr.shape, xvl.shape,ytr.shape, yvl.shape
    lmm=linear_model.LogisticRegression().fit(xtr,ytr)
    pred_y=lmm.predict(xte)
    Acc1=metrics.accuracy_score(pred_y,yte)
    print('Accuracy_score',Acc1)


for k in range(3,8):
    model=model_selection.StratifiedKFold(n_splits=k,random_state=k,shuffle=False)

    i=1
    for train_index ,test_index in model.split(X2,Y2):
        print('\n{} of fold {}'.format(i, model.n_splits))
        xtr, xte = X2.loc[train_index],X2.loc[test_index]
        ytr, yte = Y2.loc[train_index],Y2.loc[test_index]
    
        #xtr.shape, xvl.shape,ytr.shape, yvl.shape
        lmm=linear_model.LogisticRegression().fit(xtr,ytr)
        pred_y=lmm.predict(xte)
        Acc1=metrics.accuracy_score(pred_y,yte)
        print('Accuracy_score',Acc1)


###############################################################################
list(X1)

X3=X1
Y3=Y2

from sklearn import preprocessing

label=preprocessing.LabelEncoder()
X3.Sex=label.fit_transform(X3.Sex)
X3.Chol_Status=label.fit_transform(X3.Chol_Status)
X3.BP_Status=label.fit_transform(X3.BP_Status)
X3.Weight_Status=label.fit_transform(X3.Weight_Status)
X3.Smoking_Status=label.fit_transform(X3.Smoking_Status)

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth=10).fit(X3,Y3)
tree.score(X3,Y3)

T=[]
for i in range(1,15):
    t=DecisionTreeClassifier(max_depth=i).fit(X3,Y3)
    T.append(t.score(X3,Y3)*100)
print('Accuracy',max(T))

TT=[]
for i in range(1,10):
    tt=DecisionTreeClassifier(max_depth=i,criterion='entropy').fit(X3,Y3)
    TT.append(tt.score(X3,Y3)*100)
print('Accuracy',max(TT))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def find_best_model_using_gridsearchcv(X,y):
    algos={'random_forest':{
            'model':RandomForestClassifier(),
            'params':{
                    'criterion':['gini','entropy'],
                    'max_features':[5,6,7]
                    }
            },
            'logistic_Regression':{
                'model':LogisticRegression(),
                'params':{
                        
                    }
                },
                'lasso':{
                    'model':Lasso(),
                    'params':{
                            'alpha':[0.01,0.001,0.05,0.1,0.2],
                            'selection':['random','cyclic']
                    }
                },
                'decision_tree':{
                    'model':DecisionTreeClassifier(),
                    'params':{
                            'max_depth':[1,2,4,6,8,10,15,18,20,25,27,29,30],
                            'criterion' : ['gini','entropy']
                    }
                }
            }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=4)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
                })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
                    
find_best_model_using_gridsearchcv(X3,Y3)









































