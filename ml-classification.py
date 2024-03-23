import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
logr=LogisticRegression()
df=pd.read_csv("C:/Users/Admin/Downloads/archive/Iris.csv")
x=df.drop('Id',axis=1)
x=df.drop('Species',axis=1)
y=df['Species']
nb=GaussianNB()
knn=KNeighborsClassifier(n_neighbors=5)
dt=tree.DecisionTreeClassifier()
rf=RandomForestClassifier()
gbm=GradientBoostingClassifier(n_estimators=10)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)
logr.fit(X_train,Y_train)
nb.fit(X_train,Y_train)
knn.fit(X_train,Y_train)
dt.fit(X_train,Y_train)
rf.fit(X_train,Y_train)
gbm.fit(X_train,Y_train)
y_pred=logr.predict(X_test)
y_pred1=nb.predict(X_test)
y_pred2=knn.predict(X_test)
y_pred3=dt.predict(X_test)
y_pred4=rf.predict(X_test)
y_pred5=gbm.predict(X_test)
print("Naive Bayes:",accuracy_score(Y_test,y_pred))
print("K Neighbors:",accuracy_score(Y_test,y_pred2))
print("Decision Tree:",accuracy_score(Y_test,y_pred3))
print("Random Forest:",accuracy_score(Y_test,y_pred4))
print("Gradient Boosting:",accuracy_score(Y_test,y_pred5))
#print(classification_report(Y_test,y_pred))
#print(confusion_matrix(Y_test,y_pred))