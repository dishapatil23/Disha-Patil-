'''import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("C:/Users/Admin/Downloads/archive (1)/boston.csv")
print(df)
df.replace('?',"disha")
print(df.isnull().sum())
df['CRIM']=pd.cut(df['CRIM'],3,labels=['0','1','2'])
df['ZN']=pd.cut(df['ZN'],3,labels=['0','1','2'])
df['INDUS']=pd.cut(df['INDUS'],3,labels=['0','1','2'])
df['CHAS']=pd.cut(df['CHAS'],3,labels=['0','1','2'])
print(df)
X=df.drop('CHAS',axis=1)
X=X.drop('CRIM',axis=1)
Y=df['CRIM']
print(Y)
le=LabelEncoder()
le.fit(Y)
Y=le.transform(Y)
print(Y)
sns.boxplot(y='INDUS', data=df)
plt.title("Box Plot showing the distribution of sepal length")
plt.show()'''
'''Q1 = df['AGE'].quantile(0.25)
Q3 = df['AGE'].quantile(0.75)
IQR = Q3-Q1
print(IQR)
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
print(upper)
print(lower)
out1=df[df['AGE']<lower].values
out2=df[df['AGE']>upper].values
df['AGE'].replace(out1,lower,)
df['AGE'].replace(out2,upper,)'''



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df=pd.read_csv("C:/Users/Admin/Downloads/archive (1)/boston.csv")

reg=LinearRegression()


x=df.drop('MEDV',axis=1)
y=df['MEDV']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)
train=reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
score=mean_squared_error(Y_test,Y_pred)
print(score)