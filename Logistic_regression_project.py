'''
Logistic Regression Portfolio Project
advertising Data
'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def gather_data(file):
    ad_data = pd.read_csv(file)
    print(ad_data.head())
    return ad_data

def explore_data(data_frame):
    print(data_frame.info())
    print(data_frame.describe())

    plt.hist(data_frame['Age'], bins = 50)
    plt.show()

    sns.jointplot(data_frame['Age'], data_frame['Area Income'], data = data_frame)
    plt.show()

    sns.jointplot(data_frame['Age'], data_frame['Daily Time Spent on Site'], data =data_frame, kind = 'kde')
    plt.show()

    sns.jointplot(data_frame['Daily Time Spent on Site'], data_frame['Daily Internet Usage'], data = data_frame)
    plt.show()

    sns.pairplot(data_frame, hue = 'Clicked on Ad', diag_kind= 'scatter')
    plt.show()


def split_data(data_frame):
    X = data_frame[['Age', 'Area Income', 'Daily Time Spent on Site','Daily Internet Usage', 'Male']]
    y = data_frame['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    return logmodel

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

if __name__ == '__main__':
    filename = 'advertising.csv'
    ad_data = gather_data(filename)
    explore = input('Do you want to explore the data further?')
    if explore == 'yes':
        explore_data(ad_data)
    data = split_data(ad_data)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)