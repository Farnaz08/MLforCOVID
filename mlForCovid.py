import sklearn.datasets as datasets
import pandas as pd
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('Downloads/covid.csv')

X =dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)
print(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

accuracy_score(y_test, predictions)
print(predictions)
print(y_test)

accuracy_score(y_test, predictions)
