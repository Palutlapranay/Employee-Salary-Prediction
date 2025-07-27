import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("salary_data.csv")

X = data[['Experience']]
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'salary_model.pkl')
print("Model saved as salary_model.pkl")
