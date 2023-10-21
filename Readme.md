# Analysis of E-commerce dataset

In this portfolio task, we are to 
In this portfolio task, we have to build and evaluate predictive models to predict whether a user likes (rating 1) or dislikes (rating 0) an item in a dataset. The key steps in this analysis are as follows:

# Data Exploration 

All the necessary libraries are imported and the dataset is loaded. The libraries required are: 

```python 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
```

After the dataset is loaded, then we convert all the object features of the dataset into the digit features using 'OrdinalEncoder', and to understand the dataset we use 'head()' function. 

```python
encoder = OrdinalEncoder()
df[['review', 'item', 'gender', 'category']] = encoder.fit_transform(df[['review', 'item', 'gender', 'category']])
```

# Correlation of different variables

Correlation of different variables are calcualted to study the relationship and the impact of one variable with the another variable. Correlation is studied to detect the presence and strenth of a relationship between the variables. 

```python
df.corr()
```
# Splitting and training the dataset

The dataset is then splitted into training and testing sets. 
Then we print the values of testing and training sets;

```python
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
```

Training the dataset to perform the logistic regression to find the accuracy of the dataset.

# RFE feature selection

We use Recursive Feature Elimination(RFE) method to select only the most relevant features to improve the accuracy of the test. RFE removes less important features and creates a subset that maximizes predictive accuracy. 

```python
selector = RFE(train, n_features_to_select=3)
selector = selector.fit(x_train, y_train)
```

# Logistic Regression with trained features

We train logistic regression model with the selected features and evaluate the accuracy of this model. 

```python
x_train, x_test, y_train, y_test = train_test_split(df[["item", "gender", "category"]], df['rating'], test_size=0.2, random_state=42)
train = LogisticRegression()
train.fit(x_train, y_train)
y_pred = train.predict(x_test)
print("The accuracy on the test set:", accuracy_score(y_test, y_pred))
```

# K-Nearest Neighbrs(KNN)model

KNN model is then used to predict rating based on the other features and evaluate the accuracy of the model. It classifies the datapoints on how its neighbor is classfied. 

```python
neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
print("KNN accuracy on the test set is:", accuracy_score(y_test, y_pred))
```
# Hyperparameter Tuning for KNN

We then tune the hyperparameter K in KNN to see how it influences prediction performance. 

```python
parameters = {'n_neighbors': range(1, 100)}
train = GridSearchCV(neigh, parameters)
train.fit(x_train, y_train)
print('Best K value:', train.best_params_)
print('Best accuracy with optimal K value:', train.best_score_)
```

# Conclusion

We used both logistic regression and KNN model for the analysis. Logistic regression have 63% accuracy which then increased slightly to 64% after feature selection with RFE. KNN model was also applied and this gave different accuracy result with 67%. 