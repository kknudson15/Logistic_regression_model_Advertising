'''
Logistic Regression Portfolio Project
advertising Data
'''

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


ad_data = pd.read_csv('advertising.csv')
#print(ad_data.head())
#print(ad_data.info())
#print(ad_data.describe())

'''
Exploratory Data Analysis
Create a histogram of the Age
Create a joinplot showing Area income versus Age
Create a Joinplot sjowing the kde distributions of Daily time spent on site vs. Age
Create a joinplot of Daily Time Spent on Site vs. Daily Internet Usage
Create a pairplot with the hue defined by the 'Clicked on Ad' column Feature
'''
plt.hist(ad_data['Age'], bins = 50)
plt.show()

sns.jointplot(ad_data['Age'], ad_data['Area Income'], data = ad_data)
plt.show()

sns.jointplot(ad_data['Age'], ad_data['Daily Time Spent on Site'], data = ad_data, kind = 'kde')
plt.show()

sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'], data = ad_data)
plt.show()

sns.pairplot(ad_data, hue = 'Clicked on Ad', diag_kind= 'scatter')
plt.show()


'''
Logistic Regression
Split the data into training set and testing set using train_test_split
Train and fit a Logistic Regression Model on the training set
'''

X = ad_data[['Age', 'Area Income', 'Daily Time Spent on Site','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


'''
Predictions and Evaluations
Predict values for the testing data
Create a classification report for the model
'''
predictions = logmodel.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))





