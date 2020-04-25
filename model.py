import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('insurance.csv')
dummy_sex = pd.get_dummies(df['sex'])
dummy_smoker = pd.get_dummies(df['smoker'])
dummy_region = pd.get_dummies(df['region'])

df = pd.concat([df, dummy_sex, dummy_region, dummy_smoker, dummy_region], axis=1)
df.drop(['region', 'southwest', 'sex', 'smoker'], axis=1, inplace=True)

# now lets analyse
# plt.scatter(df['age'], df['charges'])
# plt.scatter(df['smoker'],df['charges'])
# plt.show()
# here we get that age is made impact on our regression
# analysis contains that each x is how affect y or how it is related to y

x = df.drop('charges', axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)


def minmax(x_train, x_test):
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    return x_train, x_test


def std(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    return x_train, x_test


x_train, x_test = std(x_train, x_test)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))

model_ridge = Ridge(alpha=0.5)
model_ridge.fit(x_train, y_train)
# print(model_ridge.score(x_test, y_test))
# print(model_ridge.score(x_train, y_train))

# lets calculate r_squared
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2)
