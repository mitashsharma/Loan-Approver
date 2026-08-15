import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/train_loan.csv")

# Features and target
x = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]


# -----------------------------
# Missing value handling
# -----------------------------

x["Gender"] = x["Gender"].fillna("Male")
x["Married"] = x["Married"].fillna("Yes")
x["Dependents"] = x["Dependents"].fillna("0")
x["Self_Employed"] = x["Self_Employed"].fillna("No")

x["LoanAmount"] = x["LoanAmount"].fillna(
    np.mean(x["LoanAmount"])
)

x["Loan_Amount_Term"] = x["Loan_Amount_Term"].fillna(
    x["Loan_Amount_Term"].mean()
)

x["Credit_History"] = x["Credit_History"].fillna(1.0)


# -----------------------------
# One-hot encoding
# -----------------------------

x = pd.get_dummies(x)


# -----------------------------
# Scaling
# Same as your notebook
# -----------------------------

mx = MinMaxScaler(feature_range=(0, 1))

scale_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount"
]

x[scale_columns] = mx.fit_transform(
    x[scale_columns]
)


# -----------------------------
# Train model
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20
)


logR = LogisticRegression(max_iter=1000)

logR.fit(x_train, y_train)


# -----------------------------
# Accuracy
# -----------------------------

prediction = logR.predict(x_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("--------------------------------")
print("Loan Approval Model")
print("--------------------------------")
print("Accuracy:", accuracy)
print("Classes:", logR.classes_)
print("Number of features:", len(x.columns))


# -----------------------------
# Save everything
# -----------------------------

model_data = {

    "model": logR,

    "scaler": mx,

    "columns": x.columns.tolist(),

    "scale_columns": scale_columns

}


joblib.dump(
    model_data,
    "model/loan_model.pkl"
)


print("--------------------------------")
print("Model saved successfully!")
print("--------------------------------")