import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

# DATA FOR PRED
data=pd.read_csv("diabetes.csv")
print(data.head())


logreg=LogisticRegression()



X=data.iloc[:,:8]
print(X.shape[1])


y=data[["Outcome"]]

X=np.array(X)
y=np.array(y)

logreg.fit(X,y.reshape(-1,))
dump(logreg,"model1.joblib")

