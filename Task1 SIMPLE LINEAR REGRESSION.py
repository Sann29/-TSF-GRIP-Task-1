#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation:Data Science and Business Analytics Internship
# 
# 
# Task-1: Prediction using Supervised Machine Learning.
# 
# Problem Statement: Predict percentage of the student on the basis of number of hours studied using Linear Regression Algorithm.
# 
# Author:Shaikh Saniya Ayub.
# 
# 

# In[1]:


#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Data

# In[4]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data.head()


# # Familiarizing with Data

# In[8]:


#Shape of dataframe

data.shape


# In[9]:


#Listing the features of the dataset

data.columns


# In[10]:


#Information about the dataset

data.info()


# In[11]:


#checking number of null value

data.isna().sum()


# In[12]:


# describtion of dataset

data.describe()


# In[13]:


#correlation betwwn data

data.corr()


# # Visualizing the data

# In[14]:


#Histogram for data visualization

data.hist(bins = 20,figsize = (15,5));


# In[15]:


#Correlation heatmap

plt.figure(figsize=(7,5))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[16]:


#scatter plot for visualization

plt.scatter(data["Hours"],data["Scores"])
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores");


# In[18]:


#spliting input and output 

y=data["Scores"]
X=data.drop("Scores",axis=1)


# # Splitting the Data

# In[19]:


# Splitting the dataset into train and test sets: 80-20 split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape


# # Model Building & Training

# In[20]:


# Linear regression model 
from sklearn.linear_model import LinearRegression

# instantiate the model
lr = LinearRegression()

# fit the model 
lr.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_lr = lr.predict(X_test)
y_train_lr = lr.predict(X_train)


# In[21]:


#computing the accuracy of the model performance
acc_train_lr = lr.score(X_train, y_train)
acc_test_lr = lr.score(X_test, y_test)


# In[22]:


#importing required libraries 
from sklearn.metrics import mean_squared_error

#computing root mean squared error (RMSE)
rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_train_lr))
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_lr))

print("Linear Regression: Accuracy on training Data: {:.3f}".format(acc_train_lr))
print("Linear Regression: Accuracy on test Data: {:.3f}".format(acc_test_lr))
print('\nLinear Regression: The RMSE of the training set is:', rmse_train_lr)
print('Linear Regression: The RMSE of the testing set is:', rmse_test_lr)


# In[23]:


#checking for actual vs predicted value

dict={"Actual":y_train,"Predicted":y_train_lr}
new_data=pd.DataFrame(dict)
new_data=new_data.reset_index(drop=1)
new_data.head()


# In[24]:


plt.scatter(X_train,y_train,label="Actual Value",color="blue")
plt.plot(X_train,y_train_lr,label="Predicted Value",color="green")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores (Training Data)")
plt.grid(True)
plt.legend();


# In[25]:


#checking for actual vs predicted value

dict={"Actual":y_test,"Predicted":y_test_lr}
new_data=pd.DataFrame(dict)
new_data=new_data.reset_index(drop=1)
new_data.head()


# In[26]:


plt.scatter(X_test,y_test,label="Actual Value",color="blue")
plt.plot(X_train,y_train_lr,label="Predicted Value",color="green")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores (Testing Data)")
plt.grid(True)
plt.legend();


# In[27]:


#predicted score if a student studies for 9.25 hrs/ day

result=lr.predict([[9.25]])
print("The Predicted score of student comes to be {:.3f} if a student studies for 9.25 hrs/ day.".format(result[0]))


# # Conclusion:
# The final take away form this project is the working of Linear Regression model on a dataset and understanding their parameters. Creating this notebook helped me to learn a lot about the parameters of the models. Accuracy of model comes to be 94.5% and the Predicted score of student comes to be 93.692 if a student studies for 9.25 hrs/ day.

# In[ ]:




