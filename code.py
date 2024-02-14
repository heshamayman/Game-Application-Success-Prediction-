import re
import pandas as pd
import numpy as np
import random
import seaborn as sns
import warnings
import time

from sklearn.svm import SVC
from sklearnex import patch_sklearn

patch_sklearn()
from matplotlib import pyplot as plt, pyplot
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from datetime import date
from pickle import dump

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'


def preprocessing(x_train):
    x_train['words in description'] = x_train['Description'].apply(lambda x: len(re.findall('(\w+)', str(x))))
    x_train.drop(columns='Description', inplace=True)

    x_train['Subtitle'] = x_train['Subtitle'].fillna('')
    x_train['Subtitle'] = x_train['Subtitle'].apply(lambda x: 0 if x == '' else 1)

    x_train['Languages'] = x_train['Languages'].fillna('')
    x_train['Languages'] = x_train['Languages'].apply(lambda x: len(x.split(', ')))

    x_train['In-app Purchases'] = x_train['In-app Purchases'].fillna('0.0')
    x_train['In-app Purchases'] = x_train['In-app Purchases'].apply(
        lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    # x_train['In-app Purchases'] = x_train['In-app Purchases'].apply(lambda x: x.split(', '))

    # Prices range from 0 to 99.99. Split into 4 categories
    # x_train['In-App-Q1'] = x_train['In-app Purchases'].apply(lambda x: 1 if any(float(i) < 25 for i in x) else 0)
    # x_train['In-App-Q2'] = x_train['In-app Purchases'].apply(lambda x: 1 if any(25 <= float(i) < 50 for i in x) else 0)
    # x_train['In-App-Q3'] = x_train['In-app Purchases'].apply(lambda x: 1 if any(50 <= float(i) < 75 for i in x) else 0)
    # x_train['In-App-Q4'] = x_train['In-app Purchases'].apply(lambda x: 1 if any(75 <= float(i) < 100 for i in x) else 0)

    genre_tr = x_train['Genres'].str.get_dummies(sep=', ')
    genre_tr = genre_tr.drop(columns='Games')
    x_train = x_train.merge(genre_tr, left_index=True, right_index=True)
    x_train = x_train.drop(columns=['Genres'])

    x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
    x_train['Age Rating'] = x_train['Age Rating'].astype(int)
    print(x_train['Age Rating'])

    # age = x_test['Age Rating'].str.get_dummies(sep=', ')
    # x_test = x_test.merge(age, left_index=True, right_index=True)
    print(x_train.columns)
    return x_train


def data_clean(x, fs, feature_selector, dev_lst, c):
    x['words in description'] = x['Description'].apply(lambda x: len(re.findall('(\w+)', str(x))))
    x.drop(columns='Description', inplace=True)

    x['Subtitle'] = x['Subtitle'].fillna('')
    x['Subtitle'] = x['Subtitle'].apply(lambda x: 0 if x == '' else 1)

    x['Languages'] = x['Languages'].fillna('')
    x['Languages'] = x['Languages'].apply(lambda x: len(x.split(', ')))

    x['In-app Purchases'] = x['In-app Purchases'].fillna('0.0')
    x['In-app Purchases'] = x['In-app Purchases'].apply(
        lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    print(x['In-app Purchases'])
    # x['In-app Purchases'] = x['In-app Purchases'].apply(lambda x: x.split(', '))

    # x['In-App-Q1'] = x['In-app Purchases'].apply(lambda x: 1 if any(float(i) < 25 for i in x) else 0)
    # x['In-App-Q2'] = x['In-app Purchases'].apply(lambda x: 1 if any(25 <= float(i) < 50 for i in x) else 0)
    # x['In-App-Q3'] = x['In-app Purchases'].apply(lambda x: 1 if any(50 <= float(i) < 75 for i in x) else 0)
    # x['In-App-Q4'] = x['In-app Purchases'].apply(lambda x: 1 if any(75 <= float(i) < 100 for i in x) else 0)

    genre_tst = x['Genres'].str.get_dummies(sep=', ')
    genre_tst = genre_tst.drop(columns='Games')
    x = x.drop(columns=['Genres'])
    x = x.merge(genre_tst, left_index=True, right_index=True)
    x = x.reindex(columns=fs, fill_value=0)
    # for col in x_tr_dr_col:
    #     x.drop(columns=col, inplace=True)
    # print(x.shape)

    x['Age Rating'] = x['Age Rating'].str.replace('+', '', regex=False)
    x['Age Rating'] = x['Age Rating'].astype(int)

    for i in range(0, x.shape[0]):
        if x['Developer'].iloc[i] not in dev_lst:
            dev_lst[x['Developer'].iloc[i]] = c
            c += 1
            x['Developer'].iloc[i] = c
        else:
            x['Developer'].iloc[i] = dev_lst[x['Developer'].iloc[i]]
    x['Developer'] = x['Developer'].astype(int)
    x = x.loc[:, feature_selector.get_support()]
    return x


def outlier_removal(X_train, y_train):
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X_train.iloc[mask, :], y_train.iloc[mask]
    return X_train, y_train


def adjust_date_columns(df):
    d = pd.DataFrame(columns=['Date'], index=range(0, df.shape[0]))
    d['Date'].fillna(date(2020, 1, 1), inplace=True)
    d['Date'] = pd.to_datetime(d['Date'])
    df['Days Since Release'] = (d['Date'] - df['Original Release Date']).dt.days
    df.drop(columns='Original Release Date', inplace=True)
    df['Days Since Last Update'] = (d['Date'] - df['Current Version Release Date']).dt.days
    df.drop(columns='Current Version Release Date', inplace=True)


# Read the data into a pandas dataframe
df = pd.read_csv('games-regression-dataset.csv', parse_dates=['Original Release Date', 'Current Version Release Date'])
cls_df = pd.read_csv('games-classification-dataset.csv',
                     parse_dates=['Original Release Date', 'Current Version Release Date'])
df.drop(columns='Primary Genre', inplace=True)

# calculating days since x
adjust_date_columns(df)

# dropping dupes + entirely unique columns as they don't contribute to pattern recognition
cls_df.drop_duplicates(inplace=True)
df.drop_duplicates(inplace=True)
df_col = list(df.columns)
unique = []
for i in range(0, df.shape[1]):
    if df.iloc[:, i].is_unique:
        unique.append(i)
for i in range(0, len(unique)):
    df.drop(columns=df_col[unique[i]], inplace=True)

X = df.drop(columns='Average User Rating')
Y_REG = pd.DataFrame()
Y_REG['Average User Rating'] = df['Average User Rating']
Y_CLS = pd.DataFrame()
Y_CLS['Rate'] = cls_df['Rate']

X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(X, Y_REG, Y_CLS, test_size=0.2,
                                                                                     random_state=0)

# iterating over developer to manually encode it
dev_lst = {'Developer': 'Label'}
c = 0
for i in range(0, X_train.shape[0]):
    if X_train['Developer'].iloc[i] not in dev_lst:
        dev_lst[X_train['Developer'].iloc[i]] = c
        c += 1
        X_train['Developer'].iloc[i] = c
    else:
        X_train['Developer'].iloc[i] = dev_lst[X_train['Developer'].iloc[i]]
X_train['Developer'] = X_train['Developer'].astype(int)

# train preprocessing
X_train = preprocessing(X_train)

print("BEFORE OUTLIER")
print(X_train.shape, y_train_reg.shape)
outlier_removal(X_train, y_train_reg)
print("AFTER OUTLIER REMOVAL")
print(X_train.shape, y_train_reg.shape)

# dropping columns that consist of 1 value entirely i.e. all 1's or all 0's in x train
X_train = X_train[[j for j in X_train if X_train[j].nunique() > 1]]
x_tr_col = X_train.columns

# feature selection via anova
fvalue_Best = SelectKBest(f_classif, k=9)
fvalue_Best.fit(X_train, y_train_reg)
X_train = X_train.loc[:, fvalue_Best.get_support()]
print(X_train.shape)

# what are scores for the features
for i in range(len(fvalue_Best.scores_)):
    print('Feature %d: %f' % (i, fvalue_Best.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fvalue_Best.scores_))], fvalue_Best.scores_)
pyplot.show()
# print(X_train)
X_test = data_clean(X_test, x_tr_col, fvalue_Best, dev_lst, c)

# apply feature scaling to avoid any overflow
# scalar = MinMaxScaler()
# X_train = pd.DataFrame(scalar.fit_transform(X_train), columns=X_train.columns)
# X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)
# encoding = [lbl, lst]
# PIK = "encoder.dat"
# with open(PIK, "wb") as f:
#     dump(encoding, f)

# data_cleaning = [x_tr_col,fvalue_Best,dev_lst,c]
# PIK = "data cleaning.dat"
# with open(PIK, "wb") as f:
#     dump(data_cleaning, f)

# simple linear regression
slr = linear_model.LinearRegression()
slr.fit(X_train, y_train_reg)
p = slr.predict(X_test)
sns.regplot(x=y_test_reg, y=p, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None).set(
    title="Simple Linear Regression")
plt.show()
print('Mean Square Error = ', metrics.mean_squared_error(y_test_reg, p))

# print('coef = ', slr.coef_)

# polynomial regression
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))
prediction = poly_model.predict(poly_features.fit_transform(X_test))
sns.regplot(x=y_test_reg, y=prediction, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None).set(
    title="Polynomial Regression");
plt.show()
print('Mean Square Error Poly = ', metrics.mean_squared_error(y_test_reg, prediction))

# lasso regression
lasso = Lasso(alpha=10)
lasso.fit(X_train, y_train_reg)
print(lasso.score(X_train, y_train_reg))
lasso_model = lasso.predict(X_test)
sns.regplot(x=y_test_reg, y=lasso_model, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None).set(
    title="Lasso Regression");
plt.show()
print('Mean Square Error Lasso = ', metrics.mean_squared_error(y_test_reg, lasso_model))

# ridge regression
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train_reg)
print(ridge.score(X_train, y_train_reg))
ridge_model = ridge.predict(X_test)
sns.regplot(x=y_test_reg, y=ridge_model, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None).set(
    title="Ridge Regression")
plt.show()
print('Mean Square Error Ridge = ', metrics.mean_squared_error(y_test_reg, ridge_model))

# PIK = "regression models.dat"
# regresion_models = [slr, poly_model, lasso, ridge]
# with open(PIK, "wb") as f:
#     dump(regresion_models, f)


# bayes = BayesianRidge()
# bayes.fit(X_train, y_train_reg)
# model = bayes.predict(X_test)
# sns.regplot(x=y_test_reg,y=model,scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None).set(title="Lasso Regression");
# plt.show()
# print('Mean Square Error Bayesian Ridge = ', metrics.mean_squared_error(y_test_reg, model))

parameters = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
logreg = LogisticRegression()
clf = GridSearchCV(logreg,  # model
                   param_grid=parameters,  # hyperparameters
                   scoring='accuracy',  # metric for scoring
                   cv=10)
# lr_ovo = OneVsOneClassifier(clf.fit(X_train, y_train_cls))
# lr_ovr = OneVsRestClassifier(clf.fit(X_train, y_train_cls))
start_time = time.time()
clf.fit(X_train, y_train_cls)
end_time = time.time()
lr_time = end_time - start_time

# model accuracy for Logistic Regression model
accuracy = clf.score(X_test, y_test_cls) * 100

x_c = ['Train Time', 'Accuracy Score']
to_plot = [lr_time, accuracy]
plt.bar(x_c, to_plot)
plt.xlabel('Bars')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

print('OneVsRest Logistic Regression accuracy: ' + str(accuracy))

# Decision Tree
dtc = DecisionTreeClassifier()
start_time = time.time()
dtc = dtc.fit(X_train, y_train_cls)
end_time = time.time()
dt_tr_time = end_time - start_time

start_time = time.time()
y_pred = dtc.predict(X_test)
end_time = time.time()
dt_tst_time = end_time - start_time

x_c = ['Train Time', 'Test Time', 'Accuracy Score']
to_plot = [dt_tr_time, dt_tst_time, accuracy_score(y_test_cls, y_pred) * 100]
plt.bar(x_c, to_plot)
plt.xlabel('Bars')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

print("Decision Tree Accuracy = ", accuracy_score(y_test_cls, y_pred) * 100)

##Random Forest
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                             n_estimators=200, oob_score=True)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100, 200],
    'n_estimators': [10, 25, 30, 50, 100, 200]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rfc,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(X_train, y_train_cls)
rfc = grid_search.best_estimator_
print(rfc)
start_time = time.time()
rfc.fit(X_train, y_train_cls)
end_time = time.time()
rfc_tr_time = end_time - start_time

start_time = time.time()
t = rfc.predict(X_train)
end_time = time.time()
rfc_tst_time = end_time - start_time
print(accuracy_score(y_train_cls, t) * 100)
p = rfc.predict(X_test)
print(accuracy_score(y_test_cls, p) * 100)

x_c = ['Train Time', 'Test Time', 'Accuracy Score']
to_plot = [rfc_tr_time, rfc_tst_time, accuracy_score(y_test_cls, p) * 100]
plt.bar(x_c, to_plot)
plt.xlabel('Bars')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

##Ensemble Training
model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
start_time = time.time()
model.fit(X_train, y_train_cls)
end_time = time.time()
end_tr_time = end_time - start_time

predict_train = model.predict(X_train)
print('\nTarget on train data', predict_train)
accuracy_train = accuracy_score(y_train_cls, predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
start_time = time.time()
predict_test = model.predict(X_test)
end_time = time.time()
end_tst_time = end_time - start_time
print('\nTarget on test data', predict_test)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test_cls, predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)

x_c = ['Train Time', 'Test Time', 'Accuracy Score']
to_plot = [end_tr_time, end_tst_time, accuracy_test]
plt.bar(x_c, to_plot)
plt.xlabel('Bars')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

# C = 0.1
# # svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train_cls)
# lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train_cls)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, y_train_cls)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train_cls)
#
# # title for the plots
# titles = ['LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']
#
# for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
#     predictions = clf.predict(X_train)
#     predictions = np.asarray(predictions).reshape(-1, 1)
#     accuracy = accuracy_score(y_train_cls, predictions) * 100
#     print(titles[i], 'train accuracy: ', accuracy)
#
#     predictions = clf.predict(X_test)
#     predictions = np.asarray(predictions).reshape(-1, 1)
#     accuracy = accuracy_score(y_test_cls, predictions) * 100
#     print(titles[i], 'test accuracy: ', accuracy, '\n')
#
# PIK = "cls.dat"
# regresion_models = [lr_ovo,lr_ovr,dtc,lin_svc,rbf_svc,poly_svc]
# with open(PIK, "wb") as f:
#     dump(regresion_models, f)

# SVM
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
start_time = time.time()
grid.fit(X_train, y_train_cls)
end_time = time.time()
svm_tr_time = end_time - start_time

start_time = time.time()
p = grid.predict(X_test)
end_time = time.time()
svm_tst_time = end_time - start_time
print(accuracy_score(p, y_test_cls))

x_c = ['Train Time', 'Test Time', 'Accuracy Score']
to_plot = [svm_tr_time, svm_tst_time, accuracy_score(p, y_test_cls)]
plt.bar(x_c, to_plot)
plt.xlabel('Bars')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()

# PIK = "classification models.dat"
# regresion_models = [clf,dtc,rfc,model,grid]
# with open(PIK, "wb") as f:
#     dump(regresion_models, f)