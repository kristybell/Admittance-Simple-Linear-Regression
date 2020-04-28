#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[5]:


raw_data = pd.read_csv('1.03. Dummies.csv')


# In[6]:


raw_data


# In[7]:


# map attendence with Yes = 1 and No = 0


# In[8]:


data = raw_data.copy()


# In[9]:


data['Attendance'] = data['Attendance'].map({'Yes':1,'No':0})


# In[10]:


data


# In[11]:


data.describe()


# In[12]:


# 46% of the students have attended the lessons


# In[13]:


# Regression


# In[14]:


y = data['GPA']
x1 = data[['SAT', 'Attendance']]


# In[16]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[19]:


plt.scatter(data['SAT'],y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[20]:


# We have two equations with the same slope but different intercepts


# In[22]:


plt.scatter(data['SAT'],y, c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[23]:


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


# In[24]:


# StatsMOdels method which takes a data frame organized 
# and then makes predicitions


# In[25]:


# How to make predictions based on the regressions we create


# In[26]:


x


# In[27]:


# Let's analyze two students
# Bob who got 1700 on the SAT and did not attend
# Alice who got 1670 on the SAT and attended


# In[28]:


new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance':[0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data


# In[29]:


new_data.rename(index={0: 'Bob', 1:'Alice'})


# In[36]:


# fitted prediction
predictions = results.predict(new_data)
predictions


# In[37]:


predictionsdf=pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:'Bob',1:'Alice'})


# In[ ]:


# Alice scored a lower SAT score, but her attendances was higher
# and thus predicted to graduate with a higher GPA than Bob

