import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model=joblib.load("iris_model.pkl")

testdf=pd.read_csv("Iris.csv")

testdf=testdf.drop("Id",axis=1)
X=testdf.drop("Species",axis=1)
y=testdf["Species"]

pred=model.predict(X)
acc=accuracy_score(pred, y)
print("Accuracy:", acc)

#1. To check the length of the predictions and the actual labels
assert len(pred)==len(y)

# #2. Accuracy Test
assert acc >= 0.90