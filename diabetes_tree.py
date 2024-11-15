# -*- coding: utf-8 -*-
"""diabetes.tree.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

df=pd.read_csv("/content/diabetes.tree.csv")
df

x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=df['Outcome'].values
print(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

decision=DecisionTreeClassifier(max_depth=3,criterion='entropy')
decision.fit(x_train,y_train)
y_pred=decision.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("precision_score:",precision_score(y_test,y_pred))
print("Confusion_matrics:",confusion_matrix(y_test,y_pred))

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(decision,filled=True)
plt.show()

