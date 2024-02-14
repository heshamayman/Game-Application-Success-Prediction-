from datetime import date
from pickle import load
import re
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score


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

    genre_tst = x['Genres'].str.get_dummies(sep=', ')
    genre_tst = genre_tst.drop(columns='Games')
    x = x.drop(columns=['Genres'])
    x = x.merge(genre_tst, left_index=True, right_index=True)
    x = x.reindex(columns=fs, fill_value=0)
    # for col in x_tr_dr_col:
    #     x.drop(columns=col, inplace=True)
    # print(x.shape)

    x['Age Rating'] = x['Age Rating'].fillna('4+')
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
    for c in x.columns:
        x[c].fillna(x[c].mean(), inplace=True)
    print(x.columns)
    return x


def adjust_date_columns(df):
    d = pd.DataFrame(columns=['Date'], index=range(0, df.shape[0]))
    d['Date'].fillna(date(2020, 1, 1), inplace=True)
    d['Date'] = pd.to_datetime(d['Date'])
    df['Days Since Release'] = (d['Date'] - df['Original Release Date']).dt.days
    df.drop(columns='Original Release Date', inplace=True)
    df['Days Since Last Update'] = (d['Date'] - df['Current Version Release Date']).dt.days
    df.drop(columns='Current Version Release Date', inplace=True)


PIK = "data cleaning.dat"
with open(PIK, "rb") as f:
    data_cleaning = load(f)
# print(data_cleaning)

PIK = "regression models.dat"
with open(PIK, "rb") as f:
    reg_models = load(f)
print(reg_models)

PIK = "poly features.dat"
with open(PIK, "rb") as f:
    poly = load(f)
print(poly)

PIK = "classification models.dat"
with open(PIK, "rb") as f:
    clf_models = load(f)
# print(clf_models)

df = pd.read_csv("ms1-games-tas-test-v1.csv", parse_dates=['Original Release Date', 'Current Version Release Date'])
cls = pd.read_csv("ms2-games-tas-test-v1.csv",parse_dates=['Original Release Date', 'Current Version Release Date'])
X = df.drop(columns='Average User Rating')
Y_REG = pd.DataFrame()
Y_REG['Average User Rating'] = df['Average User Rating']
Y_CLS = pd.DataFrame()
Y_CLS['Rate'] = cls['Rate']

adjust_date_columns(X)
X = data_clean(X,data_cleaning[0],data_cleaning[1],data_cleaning[2],data_cleaning[3])

slr = reg_models[0]
poly_reg = reg_models[1]
lasso_reg = reg_models[2]
ridge_reg = reg_models[3]

lr = clf_models[0]
dtc = clf_models[1]
rfc = clf_models[2]
gbc = clf_models[3]
svm = clf_models[4]

p = slr.predict(X)
print('Simple Linear: ',metrics.mean_squared_error(Y_REG, p))
print(metrics.r2_score(Y_REG, p))

X_POLY = poly.fit_transform(X)
p = poly_reg.predict(X_POLY)
print('Polynomial: ',metrics.mean_squared_error(Y_REG, p))
print(metrics.r2_score(Y_REG, p))

p = lasso_reg.predict(X)
print('Lasso: ',metrics.mean_squared_error(Y_REG, p))
print(metrics.r2_score(Y_REG, p))

p = ridge_reg.predict(X)
print('Ridge: ',metrics.mean_squared_error(Y_REG, p))
print(metrics.r2_score(Y_REG, p))


p = lr.predict(X)
print('Logistic Regression: ',accuracy_score(Y_CLS,p)*100)

p = dtc.predict(X)
print('Decision Tree: ',accuracy_score(Y_CLS,p)*100)

p = rfc.predict(X)
print('Random Forest Classifier: ',accuracy_score(Y_CLS,p)*100)

p = gbc.predict(X)
print('Gradient Boost Classifier: ',accuracy_score(Y_CLS,p)*100)

p = svm.predict(X)
print('SVM: ',accuracy_score(Y_CLS,p)*100)


# # title for the plots
# titles = ['Simple Linear Regression',
#           'Polynomial Regression',
#           'Lasso Regression',
#           'Ridge Regression']
#
#
# # for i, reg in enumerate((slr, poly_reg, lasso_reg, ridge_reg)):
# #     predictions = reg.predict(X)
# #     accuracy = metrics.mean_squared_error(Y_REG, predictions)
#
# for i in clf_models:
#     p = clf_models[i].predict(X)
#     print(accuracy_score(p,Y_CLS))