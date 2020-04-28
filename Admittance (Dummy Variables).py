#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('1.03. Dummies.csv')
raw_data


# map attendence with Yes = 1 and No = 0


data = raw_data.copy()

data['Attendance'] = data['Attendance'].map({'Yes':1,'No':0})
data
data.describe()

# 46% of the students have attended the lessons

# Regression
y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(data['SAT'],y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# We have two equations with the same slope but different intercepts
plt.scatter(data['SAT'],y, c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()



plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat = 0.0017*data['SAT'] + 0.275
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
fig = plt.plot(data['SAT'], yhat, lw=3, c='#4C72B0', label ='regressionline')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# StatsMOdels method which takes a data frame organized 
# and then makes predicitions

# How to make predictions based on the regressions we create

x

# Let's analyze two students
# Bob who got 1700 on the SAT and did not attend
# Alice who got 1670 on the SAT and attended

new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance':[0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data

new_data.rename(index={0: 'Bob', 1:'Alice'})

# fitted prediction
predictions = results.predict(new_data)
predictions

predictionsdf=pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:'Bob',1:'Alice'})

# Alice scored a lower SAT score, but her attendances was higher
# and thus predicted to graduate with a higher GPA than Bob

