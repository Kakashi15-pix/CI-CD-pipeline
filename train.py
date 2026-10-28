import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df=pd.read_csv(("Iris.csv"))

df=df.drop("Id",axis=1)
X=df.drop("Species",axis=1)
y=df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=20)
model=LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print("Accuracy:", accuracy)
print("Predictions:", predictions)

joblib.dump(model, "iris_model.pkl")
print("Model saved!!!")
