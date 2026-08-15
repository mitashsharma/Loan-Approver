from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


# Load model
model_data = joblib.load("model/loan_model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
model_columns = model_data["columns"]
scale_columns = model_data["scale_columns"]


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------
        # Get form values
        # -------------------------

        gender = request.form["Gender"]
        married = request.form["Married"]
        dependents = request.form["Dependents"]
        education = request.form["Education"]
        self_employed = request.form["Self_Employed"]

        applicant_income = float(
            request.form["ApplicantIncome"]
        )

        coapplicant_income = float(
            request.form["CoapplicantIncome"]
        )

        loan_amount = float(
            request.form["LoanAmount"]
        )

        loan_amount_term = float(
            request.form["Loan_Amount_Term"]
        )

        credit_history = float(
            request.form["Credit_History"]
        )

        property_area = request.form["Property_Area"]


        # -------------------------
        # Create dataframe
        # -------------------------

        data = [[

            gender,
            married,
            dependents,
            education,
            self_employed,

            applicant_income,
            coapplicant_income,
            loan_amount,

            loan_amount_term,
            credit_history,

            property_area

        ]]


        columns = [

            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",

            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",

            "Loan_Amount_Term",
            "Credit_History",

            "Property_Area"

        ]


        newdf = pd.DataFrame(
            data,
            columns=columns
        )


        # -------------------------
        # One-hot encoding
        # -------------------------

        newdf = pd.get_dummies(newdf)


        # -------------------------
        # Scale numeric columns
        # -------------------------

        newdf[scale_columns] = scaler.transform(
            newdf[scale_columns]
        )


        # -------------------------
        # Add missing columns
        # Exactly like notebook
        # -------------------------

        for col in model_columns:

            if col not in newdf.columns:

                newdf[col] = 0


        # Remove unexpected columns
        newdf = newdf[model_columns]


        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(newdf)[0]


        # Probability
        probabilities = model.predict_proba(newdf)[0]

        class_names = model.classes_

        probability_dict = dict(
            zip(class_names, probabilities)
        )


        approved_probability = (
            probability_dict.get("Y", 0) * 100
        )


        rejected_probability = (
            probability_dict.get("N", 0) * 100
        )


        print("\n-----------------------------")
        print("NEW LOAN PREDICTION")
        print("-----------------------------")

        print("Prediction:", prediction)

        print(
            "Approval probability:",
            round(approved_probability, 2),
            "%"
        )

        print(
            "Rejection probability:",
            round(rejected_probability, 2),
            "%"
        )


        # -------------------------
        # Result
        # -------------------------

        if prediction == "Y":

            result = "Loan Approved"

            status = "approved"

        else:

            result = "Loan Not Approved"

            status = "rejected"


        return render_template(

            "index.html",

            prediction=result,

            status=status,

            probability=round(
                approved_probability,
                2
            )

        )


    except Exception as e:

        print("ERROR:", e)

        return render_template(

            "index.html",

            error=str(e)

        )


if __name__ == "__main__":

    app.run(debug=True)